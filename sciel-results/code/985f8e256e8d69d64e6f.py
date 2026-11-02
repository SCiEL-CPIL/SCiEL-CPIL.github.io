"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
DEADLINE = 140.0  # 10s margin from 150s limit

def _check_deadline():
    return (time.time() - _START_TIME) > DEADLINE

def construct():
    """
    Construct a rank-28 decomposition for 2x4 by 4x8 matrix multiplication.
    
    Strategy: Split A into two 2x2 column-blocks (A_L, A_R). For each block,
    split B into top/bottom 2x8 halves. Apply Strassen to each 2x2 * 2x8
    product, yielding 7 intermediates per (block, half) pair.
    
    Total rank = 2 * 2 * 7 = 28.
    
    The 7 Strassen products for 2x2 * 2x8 (where B has columns b00,b01,b10,b11
    each being 2-vectors):
      M1 = (a00+a11) * (b00+b11)
      M2 = (a10+a11) * b00
      M3 = a00 * (b01-b11)
      M4 = a11 * (b10-b00)
      M5 = (a00+a01) * b11
      M6 = (a10-a00) * (b00+b01)
      M7 = (a01-a11) * (b10+b11)
    
    Output reconstruction:
      c0 = M1 + M4 - M5 + M7
      c1 = M3 + M5
      c2 = M2 + M4
      c3 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # A: (2,4), B: (4,8), C: (2,8)
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]   # shape (8, 28)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (32, 28)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (16, 28)
    
    if _check_deadline():
        return {"U": U, "V": V, "W": W}
    
    # For each A-block (h=0: cols 0-1, h=1: cols 2-3)
    # and each B-row-half (t=0: rows 0-1, t=1: rows 2-3):
    #   7 Strassen terms, base index = (h*2 + t) * 7
    
    for h in range(2):       # A column block: 0=left, 1=right
        for t in range(2):   # B row half: 0=top, 1=bottom
            base = (h * 2 + t) * 7
            
            # Index helpers
            # A[r, c_local] where c_local in {0,1}, actual col = h*2 + c_local
            def a_idx(r, cl):
                return r * m + (h * 2 + cl)
            
            # B[r_local, c] where r_local in {0,1}, actual row = t*2 + r_local
            # c in {0,...,7}
            def b_idx(rl, c):
                return (t * 2 + rl) * p + c
            
            # C[r, c] where r in {0,1}, c in {0,...,7}
            # W index: c * n + r
            def c_idx(r, c):
                return c * n + r
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M1 = (a00 + a11) * (b00 + b11)
            # U: a00 += 1, a11 += 1
            # V: for all c, B[0,c] += 1, B[1,c] += 1
            # W: C[0,c] += 1, C[1,c] += 1 for all c
            col = base + 0
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            for c in range(p):
                V[b_idx(0, c)][col] += 1.0
                V[b_idx(1, c)][col] += 1.0
                W[c_idx(0, c)][col] += 1.0
                W[c_idx(1, c)][col] += 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M2 = (a10 + a11) * b00
            # U: a10 += 1, a11 += 1
            # V: B[0,c] += 1 for all c
            # W: C[1,c] += 1, C[0,c] -= 1 for all c
            col = base + 1
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            for c in range(p):
                V[b_idx(0, c)][col] += 1.0
                W[c_idx(1, c)][col] += 1.0
                W[c_idx(0, c)][col] -= 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M3 = a00 * (b01 - b11)
            # U: a00 += 1
            # V: B[1,c] += 1, B[0,c] -= 1 for all c
            # W: C[0,c] += 1, C[1,c] += 1 for all c
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            for c in range(p):
                V[b_idx(1, c)][col] += 1.0
                V[b_idx(0, c)][col] -= 1.0
                W[c_idx(0, c)][col] += 1.0
                W[c_idx(1, c)][col] += 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M4 = a11 * (b10 - b00)
            # U: a11 += 1
            # V: B[0,c] += 1, B[1,c] -= 1 for all c
            # W: C[0,c] += 1, C[1,c] += 1 for all c
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            for c in range(p):
                V[b_idx(0, c)][col] += 1.0
                V[b_idx(1, c)][col] -= 1.0
                W[c_idx(0, c)][col] += 1.0
                W[c_idx(1, c)][col] += 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M5 = (a00 + a01) * b11
            # U: a00 += 1, a01 += 1
            # V: B[1,c] += 1 for all c
            # W: C[0,c] -= 1, C[1,c] += 1 for all c
            col = base + 4
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(0, 1)][col] += 1.0
            for c in range(p):
                V[b_idx(1, c)][col] += 1.0
                W[c_idx(0, c)][col] -= 1.0
                W[c_idx(1, c)][col] += 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M6 = (a10 - a00) * (b00 + b01)
            # U: a10 += 1, a00 -= 1
            # V: B[0,c] += 1, B[1,c] += 1 for all c
            # W: C[1,c] += 1 for all c
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            for c in range(p):
                V[b_idx(0, c)][col] += 1.0
                V[b_idx(1, c)][col] += 1.0
                W[c_idx(1, c)][col] += 1.0
            
            if _check_deadline():
                return {"U": U, "V": V, "W": W}
            
            # M7 = (a01 - a11) * (b10 + b11)
            # U: a01 += 1, a11 -= 1
            # V: B[0,c] += 1, B[1,c] += 1 for all c
            # W: C[0,c] += 1 for all c
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            for c in range(p):
                V[b_idx(0, c)][col] += 1.0
                V[b_idx(1, c)][col] += 1.0
                W[c_idx(0, c)][col] += 1.0
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for s in range(rank):
                    val += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that invalid entries are 0
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = 0.0
            for s in range(rank):
                val += U[a, s] * V[b, s] * W[c, s]
            if abs(val) > 1e-9:
                return False
    
    return True
