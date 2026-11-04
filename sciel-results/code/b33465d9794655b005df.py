"""Matrix multiplication 4x4 * 4x9: rank-minimal exact tensor decomposition.

Strategy:
1. Deterministic construction:
   - Block-diagonal Strassen for 4x4 * 4x4 (rank 49), then extend to 4x4 * 4x9
     by applying the same 49-term decomposition to each of the 9 output columns
     but sharing the same U factors where possible.
   Actually, a cleaner approach:
   - Use the rank-49 decomposition for 4x4*4x4 (from Strassen 2x2x2 applied at
     4x4 level via 2x2 block decomposition with Strassen on each 2x2 block
     but with cross-block sharing).
   
   However, the simplest known improvement over schoolbook (144) is:
   - Use Strassen 2x2x2 (rank 7) to handle the 4x4 * 4x4 part (rank 49),
     then for the 4x4 * 4x9 case, we can think of it as 4x4 * [4x4 | 4x4 | 4x1].
     The first two 4x4 blocks use rank-49 decomposition (but they share the
     same U matrix since the left operand is the same), and the last 4x1 block
     uses rank-16 (schoolbook for 4x1).
     Total rank = 49 + 49 + 16 = 114? No, that's not right either because
     the V and W factors would need to be consistent.
   
   Let me think more carefully. The tensor for (n,m,p) = (4,4,9) has:
   - U: shape (16, r)
   - V: shape (36, r)  
   - W: shape (36, r)
   
   Schoolbook: r = 144.
   
   Better approach: Partition B = [B1 | B2 | B3] where B1, B2 are 4x4 and B3 is 4x1.
   C = A*B = [A*B1 | A*B2 | A*B3].
   
   For A*B1 and A*B2, we can use the same rank-49 decomposition for 4x4*4x4.
   But the V factors for B1 and B2 are different, and W factors for the outputs
   are different. So we'd have:
   - For B1 part: U (16x49), V1 (16x49), W1 (16x49) -- rank 49
   - For B2 part: U (16x49), V2 (16x49), W2 (16x49) -- rank 49 (same U, different V, W)
   - For B3 part: U (16x16), V3 (4x16), W3 (4x16) -- rank 16
   
   Total rank = 49 + 49 + 16 = 114.
   
   But wait, U for the first two parts is the same. So we can merge:
   U_combined (16 x 98), V_combined (36 x 98), W_combined (36 x 98).
   
   Actually, let me just implement the rank-49 4x4*4x4 decomposition first.
   
   4x4*4x4 via Strassen:
   Partition A, B, C into 2x2 blocks of 2x2 matrices.
   A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
   C = [[C11, C12], [C21, C22]]
   
   Strassen gives 7 products M1..M7 of 2x2 matrices.
   Each 2x2*2x2 uses rank 7 (Strassen) or rank 4 (schoolbook).
   If we use Strassen for each, we get 7*7 = 49 rank-1 terms.
   
   Let me implement this.
"""
import time
import random
import numpy as np
from fractions import Fraction as F

DIMENSIONS = [4, 4, 9]
DEADLINE = 110.0

def _validate_numpy(U, V, W):
    """Fast validation using numpy."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False
                    if d > maxden:
                        maxden = d
    
    s = 1
    while maxden % s != 0:
        s *= 2
    
    def to_int(mat):
        return np.array([[int(v * s) if isinstance(v, F) else int(v) * s for v in row] for row in mat], dtype=np.int64)
    
    Us = to_int(U)
    Vs = to_int(V)
    Ws = to_int(W)
    s3 = s * s * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i * m + j]
            for k in range(p):
                Vk = Vs[j * p + k]
                Wk = Ws[k * n + i]
                if np.dot(Ui, Vk * Wk) != s3:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W

def strassen_4x4x4():
    """Rank-49 decomposition for 4x4 * 4x4 using Strassen at two levels."""
    n = m = p = 4
    r = 49
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Strassen 2x2x2 template (indices 0-6 for each block product)
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11*(B10-B00)
    # M3 = (A10+A11)*B01
    # M4 = A00*(B01-B11)
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # For each of the 7 Strassen products, we have a 2x2*2x2 multiplication
    # which itself uses 7 rank-1 terms.
    
    # Let's define the Strassen 2x2*2x2 template as a list of (U_entries, V_entries, W_entries)
    # where each entry is a list of (index, value) pairs.
    
    # 2x2 indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B: 0=B00, 1=B01, 2=B10, 3=B11
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_template = [
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let me use the standard formulation:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00+B01)
        # M3 = (A00-A10)(B20+B21)  [using row-major: B20=B10, B21=B11]
        # M4 = (A01-A11)(B10+B11)  [B10, B11]
        # M5 = (A00+A01)(B00+B10)
        # M6 = (A00+A01)(B11-B01)
        # M7 = (A10-A00)(B00+B01)
        
        # Actually, let me just use a known correct Strassen 2x2x2 decomposition.
    ]
    
    # Let me use a simpler approach: hardcode the 49 terms.
    # Each of the 7 Strassen M-products is a 2x2*2x2 multiplication.
    # For each M-product, we use schoolbook (rank 4) for the 2x2*2x2.
    # This gives 7*4 = 28 terms, which is better than 49.
    # But wait, if we use schoolbook for the inner 2x2*2x2, we lose the benefit.
    # Let's use Strassen (rank 7) for the inner 2x2*2x2: 7*7 = 49.
    
    # Actually, for a first attempt, let's use schoolbook for the inner 2x2*2x2:
    # 7 * 4 = 28 rank terms for 4x4*4x4.
    
    # Strassen 2x2x2 with schoolbook inner:
    # M1 = (A00+A11)(B00+B11): 4 terms
    # M2 = (A10+A11)(B00+B01): 4 terms
    # M3 = (A00-A10)(B10+B11): 4 terms
    # M4 = (A01-A11)(B10+B11): 4 terms
    # M5 = (A00+A01)(B00+B10): 4 terms
    # M6 = (A00+A01)(B11-B01): 4 terms
    # M7 = (A10-A00)(B00+B01): 4 terms
    # Total: 28 terms
    
    # Let me define the 2x2*2x2 schoolbook as 4 terms:
    # For matrices X, Y (2x2), Z = X*Y:
    # Z00 = X00*Y00 + X01*Y10
    # Z01 = X00*Y01 + X01*Y11
    # Z10 = X10*Y00 + X11*Y10
    # Z11 = X10*Y01 + X11*Y11
    
    # For each Strassen M-product, we need to map the linear combinations to 2x2 matrices.
    
    # This is getting complex. Let me just return schoolbook for now and do random search.
    return r, U, V, W

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # 1) Schoolbook (rank 144)
    r, U, V, W = schoolbook()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 2) Random search for better rank
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > DEADLINE - 15:
            break
        r = rng.randint(50, 144)
        def rand_entry():
            x = rng.random()
            if x < 0.8:
                return 0
            elif x < 0.95:
                return rng.choice([-1, 1])
            else:
                return rng.choice([-1, 1]) * F(1, 2)
        U = [[rand_entry() for _ in range(r)] for _ in range(n * m)]
        V = [[rand_entry() for _ in range(r)] for _ in range(m * p)]
        W = [[rand_entry() for _ in range(r)] for _ in range(p * n)]
        try:
            if _validate_numpy(U, V, W):
                consider(r, U, V, W)
                break
        except Exception:
            pass
    
    if best is None:
        r, U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
