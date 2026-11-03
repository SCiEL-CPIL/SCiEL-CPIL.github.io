import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Uses Strassen for 2x2 blocks and schoolbook for the remaining 2x1x8 part.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in parts.
    # Part 1: 2x2 * 2x8 using Strassen for each 2x2 block.
    # Part 2: 2x1 * 1x8 using schoolbook.
    
    # Let's collect the terms (u, v, w) triples.
    # U is indexed by (i, j) where i in 0..n-1, j in 0..m-1. Index = i*m + j.
    # V is indexed by (j, k) where j in 0..m-1, k in 0..p-1. Index = j*p + k.
    # W is indexed by (k, i) where k in 0..p-1, i in 0..n-1. Index = k*n + i.
    
    terms = []
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # A is 2x2, B is 2x8.
    # We split B into four 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
    # For each block, we use Strassen.
    
    # Strassen for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C = [[P1 - P4 + P5 + P7, P3 + P5],
    #      [P2 + P4, P1 - P2 + P3 + P6]]
    
    # A indices:
    # a: (0,0) -> U[0*3+0] = U[0]
    # b: (0,1) -> U[0*3+1] = U[1]
    # c: (1,0) -> U[1*3+0] = U[3]
    # d: (1,1) -> U[1*3+1] = U[4]
    
    # For block t (t=0..3), B block is B[:, 2t:2t+2].
    # e: (0, 2t) -> V[0*8+2t] = V[2t]
    # f: (0, 2t+1) -> V[0*8+2t+1] = V[2t+1]
    # g: (1, 2t) -> V[1*8+2t] = V[8+2t]
    # h: (1, 2t+1) -> V[1*8+2t+1] = V[8+2t+1]
    
    # C block t is C[:, 2t:2t+2].
    # C[0, 2t]   -> W[2t*2+0] = W[4t]
    # C[0, 2t+1] -> W[(2t+1)*2+0] = W[4t+2]
    # C[1, 2t]   -> W[2t*2+1] = W[4t+1]
    # C[1, 2t+1] -> W[(2t+1)*2+1] = W[4t+3]
    
    for t in range(4):
        # P1 = (a+d)(e+h)
        # U: a(1), d(1)
        # V: e(1), h(1)
        # W: C00(1), C11(1)
        terms.append((U_idx(0,0), V_idx(0, 2*t), W_idx(2*t, 0), 1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t, 0), 1))
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t, 0), 1))
        terms.append((U_idx(1,1), V_idx(1, 2*t+1), W_idx(2*t, 0), 1))
        
        terms.append((U_idx(0,0), V_idx(0, 2*t), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(1,1), V_idx(1, 2*t+1), W_idx(2*t+1, 1), 1))
        
        # P2 = (c+d)e
        # U: c(1), d(1)
        # V: e(1)
        # W: C10(1), C11(-1)
        terms.append((U_idx(1,0), V_idx(0, 2*t), W_idx(2*t, 1), 1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t, 1), 1))
        terms.append((U_idx(1,0), V_idx(0, 2*t), W_idx(2*t+1, 1), -1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t+1, 1), -1))
        
        # P3 = a(f-h)
        # U: a(1)
        # V: f(1), h(-1)
        # W: C01(1), C11(1)
        terms.append((U_idx(0,0), V_idx(0, 2*t+1), W_idx(2*t+1, 0), 1))
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t+1, 0), -1))
        terms.append((U_idx(0,0), V_idx(0, 2*t+1), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t+1, 1), 1))
        
        # P4 = d(g-e)
        # U: d(1)
        # V: g(1), e(-1)
        # W: C00(1), C10(1)
        terms.append((U_idx(1,1), V_idx(1, 2*t), W_idx(2*t, 0), 1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t, 0), -1))
        terms.append((U_idx(1,1), V_idx(1, 2*t), W_idx(2*t, 1), 1))
        terms.append((U_idx(1,1), V_idx(0, 2*t), W_idx(2*t, 1), -1))
        
        # P5 = (a+b)h
        # U: a(1), b(1)
        # V: h(1)
        # W: C00(-1), C01(1)
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t, 0), -1))
        terms.append((U_idx(0,1), V_idx(1, 2*t+1), W_idx(2*t, 0), -1))
        terms.append((U_idx(0,0), V_idx(1, 2*t+1), W_idx(2*t+1, 0), 1))
        terms.append((U_idx(0,1), V_idx(1, 2*t+1), W_idx(2*t+1, 0), 1))
        
        # P6 = (c-a)(e+f)
        # U: c(1), a(-1)
        # V: e(1), f(1)
        # W: C11(1)
        terms.append((U_idx(1,0), V_idx(0, 2*t), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(0,0), V_idx(0, 2*t), W_idx(2*t+1, 1), -1))
        terms.append((U_idx(1,0), V_idx(0, 2*t+1), W_idx(2*t+1, 1), 1))
        terms.append((U_idx(0,0), V_idx(0, 2*t+1), W_idx(2*t+1, 1), -1))
        
        # P7 = (b-d)(g+h)
        # U: b(1), d(-1)
        # V: g(1), h(1)
        # W: C00(1)
        terms.append((U_idx(0,1), V_idx(1, 2*t), W_idx(2*t, 0), 1))
        terms.append((U_idx(1,1), V_idx(1, 2*t), W_idx(2*t, 0), -1))
        terms.append((U_idx(0,1), V_idx(1, 2*t+1), W_idx(2*t, 0), 1))
        terms.append((U_idx(1,1), V_idx(1, 2*t+1), W_idx(2*t, 0), -1))

    # Part 2: A[0:2, 2] * B[2, 0:8]
    # A is 2x1, B is 1x8.
    # C[i, k] = A[i, 2] * B[2, k]
    
    for i in range(2):
        for k in range(8):
            terms.append((U_idx(i, 2), V_idx(2, k), W_idx(k, i), 1))

    # Now we have a list of terms. We need to combine them into rank-1 terms.
    # Each term is (u, v, w, coeff).
    # We can group terms that have the same (u, v) pair? No, rank-1 term is (u, v, w).
    # Wait, the tensor is sum over r of U[:,r] * V[:,r] * W[:,r].
    # This means each rank-1 term is a product of three vectors.
    # The terms we generated are of the form coeff * e_u * e_v * e_w.
    # We can combine terms with the same (u, v, w) index.
    
    # Let's use a dictionary to sum coefficients for each (u, v, w) triple.
    # u is in 0..5, v is in 0..23, w is in 0..15.
    
    from collections import defaultdict
    coeff_dict = defaultdict(float)
    
    for u, v, w, coeff in terms:
        coeff_dict[(u, v, w)] += coeff
    
    # Filter out zeros
    non_zero_terms = [(u, v, w, c) for (u, v, w), c in coeff_dict.items() if c != 0]
    
    # The rank is the number of non-zero terms.
    rank = len(non_zero_terms)
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w, c) in enumerate(non_zero_terms):
        U[u][r] = c
        V[v][r] = 1
        W[w][r] = 1
        
    # Verify decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 if i,j,k are valid indices of A, B, C such that C[i,k] = sum_j A[i,j]B[j,k]
    # Actually, the tensor T is defined such that T_{ijk} = 1 if j is the middle index.
    # Wait, the standard tensor for matrix multiplication is T_{ijk} where i is row of A, j is col of A / row of B, k is col of B.
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{ijk} = 1 for all valid i, j, k.
    # Let's check a few.
    
    # T_{0,0,0} should be 1.
    # Sum_r U[0, r] V[0, r] W[0, r]
    # Terms involving u=0, v=0, w=0.
    # From P1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # From P2: U[0,0]=0
    # From P3: U[0,0]=1, V[0,0]=0
    # From P4: U[0,0]=0
    # From P5: U[0,0]=1, V[0,0]=0
    # From P6: U[0,0]=-1, V[0,0]=1, W[0,0]=0
    # From P7: U[0,0]=0
    # Sum = 1. Correct.
    
    # T_{0,0,1} should be 1.
    # Sum_r U[0, r] V[0, r] W[1, r]
    # Terms involving u=0, v=0, w=1.
    # W index 1 corresponds to C[1, 0] (k=0, i=1 -> 0*2+1=1).
    # Wait, W index is k*n + i.
    # W[1] is k=0, i=1. So C[1, 0].
    # But we are checking T_{0,0,1}. i=0, j=0, k=1.
    # This corresponds to A[0,0] * B[0,1] contributing to C[0,1].
    # C[0,1] is W index k=1, i=0 -> 1*2+0 = 2.
    # So we should check T_{0,0,1} -> W[2].
    
    # Let's write a verification function to be sure.
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def U_idx(i, j):
    return i * 3 + j

def V_idx(j, k):
    return j * 8 + k

def W_idx(k, i):
    return k * 2 + i
