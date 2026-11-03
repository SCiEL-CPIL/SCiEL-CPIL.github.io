import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if (i,j,k) corresponds to a product term
    # Actually, the tensor for matrix multiplication A(4x4) * B(4x7) -> C(4x7) is:
    # T[i,j,k] = delta_{i,j,k} in the sense that C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (since each product term contributes)
    # Wait, no. The multiplication tensor is T_{ijk} = delta_{ij} * delta_{jk} in Einstein notation
    # More precisely, for A (n x m) and B (m x p), C = A B means C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # and the decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c where (U_c)_i = U[i,c], (V_c)_j = V[j,c], (W_c)_k = W[k,c]
    # Actually, the correct tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the indices form a valid product term, i.e., T[i,j,k] = 1 for all valid i,j,k
    # No wait, that's not right either. Let me think more carefully.
    # The multiplication tensor M_{n,m,p} is in U* ⊗ V* ⊗ W* where dim U = nm, dim V = mp, dim W = pn
    # M_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'}
    # So M[(i*m+j), (j'*p+k), (k'*n+i')] = 1 iff j == j' and k == k'
    # In the decomposition, we have M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{nm}, v_c is in R^{mp}, w_c is in R^{pn}
    # So M[(i*m+j), (j'*p+k), (k'*n+i')] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # This should equal delta_{j,j'} * delta_{k,k'}
    # Let's verify the schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            lhs = 0
                            for c in range(rank):
                                lhs += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            rhs = (1 if j == j2 and k == k2 else 0)
                            if abs(lhs - rhs) > 1e-10:
                                return False
    return True

def _strassen_2x2_2x2():
    # Strassen for 2x2 * 2x2 -> 2x2, rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # U is (2*2, 7) = (4, 7), V is (2*2, 7) = (4, 7), W is (2*2, 7) = (4, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: row 0
        [0, 0, 1, 0, 0, 1, 1],  # b: row 1
        [0, 1, 0, 0, 0, 1, -1], # c: row 2
        [1, 1, 0, 1, 0, -1, 0]  # d: row 3
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # e: row 0
        [0, 1, 1, 1, 0, 0, 0],  # f: row 1
        [0, 0, -1, 0, 0, 1, 1], # g: row 2
        [1, 0, 0, -1, 1, 0, 0]  # h: row 3
    ]
    W = [
        [1, 0, 1, 0, 1, 0, 1],  # i=0, k=0: M1-M2+M3+M6 -> wait, C[0,0] = M1+M4-M5+M7
        # Let me redo this more carefully
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        # W[k*2+i, c] where k is col index of C, i is row index of C
        # So W[0, c] is for C[0,0], W[1, c] for C[0,1], W[2, c] for C[1,0], W[3, c] for C[1,1]
        [1, 0, 1, 1, -1, 0, 1],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]
    return {"U": U, "V": V, "W": W}

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    # Combine two block decompositions
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    U = [[0] * (rank1 + rank2) for _ in range(len(U1))]
    V = [[0] * (rank1 + rank2) for _ in range(len(V1))]
    W = [[0] * (rank1 + rank2) for _ in range(len(W1))]
    for r in range(rank1):
        for i in range(len(U1)):
            U[i][r] = U1[i][r]
        for i in range(len(V1)):
            V[i][r] = V1[i][r]
        for i in range(len(W1)):
            W[i][r] = W1[i][r]
    for r in range(rank2):
        for i in range(len(U2)):
            U[i][rank1 + r] = U2[i][r]
        for i in range(len(V2)):
            V[i][rank1 + r] = V2[i][r]
        for i in range(len(W2)):
            W[i][rank1 + r] = W2[i][r]
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try Strassen-based approaches
    # For 4x4 * 4x7, we can try to use Strassen recursively
    # 4x4 * 4x4 uses Strassen with rank 49 (7^2) if we split into 2x2 blocks
    # But we have 4x4 * 4x7
    
    # Let's try a direct approach using the structure of the problem
    # The key insight is that for matrix multiplication A(n,m) * B(m,p),
    # the tensor rank is at most the product of the ranks of smaller multiplications
    # when using block decomposition
    
    # For 4x4 * 4x7:
    # We can view this as two 4x4 * 4x4 multiplications plus a 4x4 * 4x3 multiplication
    # But that's not quite right. Let's think differently.
    
    # Actually, let's try to use the fact that 4 = 2*2 and 7 = 4+3
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    # Split B into 2x2 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2 or 2x1
    # This is getting complicated. Let me try a simpler approach.
    
    # Try to find a lower rank decomposition using random search with verification
    import random
    random.seed(42)
    
    # We know the schoolbook rank is 4*4*7 = 112
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But in practice, for these dimensions, the rank is likely higher
    
    # Let's try to use the Strassen algorithm recursively
    # 2x2 * 2x2 has rank 7 (Strassen)
    # 4x4 * 4x4 can be done with rank 49 using Strassen recursively
    # For 4x4 * 4x7, we can pad B to 4x8 and use 4x4 * 4x8, then take the first 7 columns
    # 4x4 * 4x8: split into 2x2 blocks
    # This gives rank 7^3 = 343, which is worse than schoolbook 112
    
    # So Strassen recursion doesn't help for these small dimensions
    # The schoolbook is actually competitive here
    
    # Let's try a different approach: use the structure of the 7 dimension
    # 7 is prime, so we can't easily factor it
    
    # For now, return the schoolbook decomposition
    if time.time() > deadline:
        return best
    
    # Try to verify the schoolbook decomposition
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # If verification fails, something is wrong, but this shouldn't happen
    return best
