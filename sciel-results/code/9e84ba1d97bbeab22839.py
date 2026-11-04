"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses structured algebraic search with exact rational verification.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(n, m, p, U, V, W):
    """
    Verify exact decomposition using Fraction arithmetic.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                    vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                    wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _verify_float(n, m, p, U, V, W, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    Each term is a triple (u, v, w) where u, v, w are 2x2 matrices (flattened to length 4).
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B where A, B are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+f)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # where the product (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # The standard tensor decomposition for 2x2x2 has rank 7.
    # Let's construct it explicitly.
    
    # Indexing: i in {0,1}, j in {0,1}, k in {0,1}
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1 = (a+d)(e+h)
    # a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
    # U[0,0]=1 (a), U[3,0]=1 (d)
    # V[0,0]=1 (e), V[3,0]=1 (h)
    # W[0,0]=1 (C[0,0]), W[3,0]=1 (C[1,1])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (c+d)e
    # c = A[1,0], d = A[1,1], e = B[0,0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C[1,0]
    
    # Term 3: M3 = a(f-h)
    # a = A[0,0], f = B[0,1], h = B[1,1]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C[0,1]
    
    # Term 4: M4 = d(g-e)
    # d = A[1,1], g = B[1,0], e = B[0,0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1  # C[1,0]
    
    # Term 5: M5 = (a+b)h
    # a = A[0,0], b = A[0,1], h = B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0] gets -1, C[0,1] gets 1
    
    # Term 6: M6 = (c-d)(f+g)
    # c = A[1,0], d = A[1,1], f = B[0,1], g = B[1,0]
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Term 7: M7 = (b-d)(e+f)
    # b = A[0,1], d = A[1,1], e = B[0,0], f = B[0,1]
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _try_strassen_5x5(n, m, p, deadline):
    """
    Try to construct a 5x5x5 decomposition using block decomposition with Strassen.
    
    Strategy: Partition 5x5 into blocks.
    One approach: 5 = 2 + 3
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    
    C = A * B:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This is just standard block multiplication, which gives rank 125.
    
    To get better, we need to use Strassen-like tricks on the blocks.
    
    Alternative: Use the fact that we can embed a 4x4x4 decomposition into 5x5x5.
    The best known rank for 4x4x4 is 49 (or lower).
    
    Let's try a different approach: use the "border rank" or "approximate rank" ideas.
    But we need exact decomposition.
    
    For now, let's just try to find a decomposition by solving a system of equations
    with rank < 125.
    
    We'll use a simple iterative approach: start with schoolbook and try to merge columns.
    """
    nm, mp, pn = n * m, m * p, p * n
    best_rank = n * m * p
    
    # Try to find a decomposition with rank < 125
    # We'll use a greedy approach: try to merge pairs of columns
    
    for target_rank in range(49, best_rank):
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((pn, target_rank), dtype=float)
        
        # Assign the 125 triples to target_rank columns
        # Use a deterministic assignment
        for c in range(target_rank):
            i = (c * 7) % n
            j = (c * 13) % m
            k = (c * 17) % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        # Check if this works
        if _verify_float(n, m, p, U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
            # Verify exactly
            U_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in U.tolist()]
            V_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in V.tolist()]
            W_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in W.tolist()]
            
            if _verify_exact(n, m, p, U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}, target_rank
    
    return None, best_rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-like
    if time.time() < deadline:
        result, rank = _try_strassen_5x5(n, m, p, deadline)
        if result is not None and rank < best_rank:
            best = result
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
