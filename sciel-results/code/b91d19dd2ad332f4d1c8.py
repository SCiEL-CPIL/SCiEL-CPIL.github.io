"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We want to decompose the matrix multiplication tensor for 2x4 * 4x8.
    # The tensor has entries T[i, j, k] = 1 for all i in [0,2), j in [0,4), k in [0,8).
    # A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all i,j,k.
    
    # Strategy: Use a Kronecker product structure.
    # 2x4 * 4x8 can be seen as combining:
    # - A 2x2 * 2x2 multiplication (Strassen, rank 7) for the "outer" dimensions
    # - A 1x2 * 2x4 multiplication (schoolbook, rank 4) for the "inner" dimensions
    # But the dimensions don't align perfectly for a simple Kronecker product.
    
    # Let's think of it differently. 
    # A is 2x4, B is 4x8.
    # We can partition A into 2 blocks of 2x2: A = [A0, A1] where A0, A1 are 2x2.
    # We can partition B into 4 blocks of 2x2: B = [B0, B1, B2, B3] where Bj are 2x2.
    # Then C = A*B = [A0*B0 + A1*B2, A0*B1 + A1*B3].
    # C is 2x8, partitioned into 4 blocks of 2x2: C = [C0, C1, C2, C3].
    # C0 = A0*B0 + A1*B2
    # C1 = A0*B1 + A1*B3
    # C2 = ... wait, this partitioning isn't standard.
    
    # Let's use the standard Kronecker product approach.
    # M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4} ? No, that's not right.
    # M_{n,m,p} ⊗ M_{n',m',p'} = M_{n*n', m*m', p*p'}
    # So M_{2,4,8} = M_{2,2,4} ⊗ M_{1,2,2} = M_{2,2,4} ⊗ M_{1,2,2}
    # Or M_{2,4,8} = M_{2,4,2} ⊗ M_{1,2,4}
    # Or M_{2,4,8} = M_{2,2,8} ⊗ M_{1,2,1}
    
    # Let's try M_{2,4,8} = M_{2,2,4} ⊗ M_{1,2,2}
    # M_{2,2,4}: multiply 2x2 by 2x4. Rank? 
    #   Schoolbook: 2*2*4 = 16.
    #   Can we do better? The tensor is T[i,j,k] where i in [0,2), j in [0,2), k in [0,4).
    #   For each k, we have a 2x2x2 tensor (M_{2,2,2}), rank 7.
    #   So M_{2,2,4} has rank 7*2 = 14? Not necessarily, because the k dimension is separate.
    #   Actually, M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2}. Rank = 7 * 2 = 14.
    #   Or M_{2,2,4} = M_{2,2,1} ⊗ M_{1,1,4}. Rank = 4 * 4 = 16.
    #   The Kronecker product gives an upper bound. The actual rank might be lower.
    #   But for now, let's use the Kronecker product bound.
    
    # M_{1,2,2}: multiply 1x2 by 2x2. Rank = 4 (schoolbook, minimal since it's a "row times matrix").
    #   Actually, M_{1,2,2} has T[i,j,k] with i=0, j in [0,2), k in [0,2).
    #   T[0,j,k] = 1. This is just the identity in some sense. Rank = 4.
    
    # So M_{2,4,8} = M_{2,2,4} ⊗ M_{1,2,2} would have rank 14 * 4 = 56.
    # But we can do better by using Strassen for M_{2,2,2} parts.
    
    # Let's use the decomposition: M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}
    # M_{2,2,2}: rank 7 (Strassen)
    # M_{1,2,4}: multiply 1x2 by 2x4. Rank = 4 (schoolbook).
    #   T[i,j,k] with i=0, j in [0,2), k in [0,4). T[0,j,k] = 1.
    #   U: (1*2, 4) = (2, 4). U[j, r] = 1 if j==r else 0.
    #   V: (2*4, 4) = (8, 4). V[j*4+k, r] = 1 if j==r else 0.
    #   W: (4*1, 4) = (4, 4). W[k, r] = 1.
    #   Check: sum_r U[j,r] V[j*4+k,r] W[k,r] = sum_r (j==r) (j==r) (1) = 1 if j in [0,2) and r=j, else 0.
    #   Wait, that's not right. The sum should be over r, and for each (j,k) we want 1.
    #   For fixed j,k: sum_r U[j,r] V[j*4+k,r] W[k,r]
    #   U[j,r] is 1 only if r=j. V[j*4+k,r] is 1 only if r=j. W[k,r] is 1 for all r.
    #   So the sum is 1 if j is in [0,2) and r=j, which gives 1. Correct.
    
    # So the Kronecker product M_{2,2,2} ⊗ M_{1,2,4} has rank 7 * 4 = 28.
    # This matches the current program's approach.
    
    # Can we do better than 28? 
    # The theoretical lower bound for M_{2,4,8} is not well-known, but 28 is a good upper bound.
    # Let's try to verify if 28 is achievable and if we can find a rank < 28.
    
    # For now, let's implement the rank-28 decomposition correctly.
    # The current program has a bug in the Kronecker product construction.
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M_{2,2,2} Strassen decomposition (rank 7)
    # i_B in {0, 1}, j_B in {0, 1}, k_B in {0, 1}
    # U_B[i_B * 2 + j_B, r_B], V_B[j_B * 2 + k_B, r_B], W_B[k_B * 2 + i_B, r_B]
    
    U_B = np.array([
        [1,  0,  1,  0,  1,  0,  0],   # i_B=0, j_B=0
        [0,  1,  0,  0,  0,  1,  0],   # i_B=1, j_B=0
        [0,  0,  0,  0,  1, -1,  1],   # i_B=0, j_B=1
        [1,  1,  0,  1,  0,  0, -1],   # i_B=1, j_B=1
    ], dtype=float)
    
    V_B = np.array([
        [1,  1,  0,  0,  0,  1,  0],   # j_B=0, k_B=0
        [0,  0,  1,  1,  0,  0,  1],   # j_B=1, k_B=0
        [0,  0,  0, -1,  0,  1,  0],   # j_B=0, k_B=1
        [1,  0, -1,  0,  1,  0,  1],   # j_B=1, k_B=1
    ], dtype=float)
    
    W_B = np.array([
        [1,  0,  0,  0,  0,  0,  0],   # k_B=0, i_B=0
        [-1, 0, 0, 1, 0, 1, 0],        # k_B=0, i_B=1
        [-1, 1, 1, 0, 0, 0, 0],        # k_B=1, i_B=0
        [1,  0, 0, 0, 1, 0, 1],        # k_B=1, i_B=1
    ], dtype=float)
    
    # M_{1,2,4} schoolbook decomposition (rank 4)
    # i_A = 0, j_A in {0, 1}, k_A in {0, 1, 2, 3}
    # U_A[j_A, r_A] = 1 if j_A == r_A else 0
    # V_A[j_A * 4 + k_A, r_A] = 1 if j_A == r_A else 0
    # W_A[k_A, r_A] = 1
    
    # Combined Kronecker product:
    # i = i_B (i_A = 0)
    # j = 2*j_A + j_B  (j_A in [0,2), j_B in [0,2))
    # k = 4*k_A + k_B  (k_A in [0,4), k_B in [0,2))
    # Wait, the dimensions are:
    #   A is 2x4: i in [0,2), j in [0,4). j = 2*j_A + j_B with j_A in [0,2), j_B in [0,2).
    #   B is 4x8: j in [0,4), k in [0,8). k = 4*k_A + k_B with k_A in [0,2), k_B in [0,4)? No.
    #   Let's re-examine the Kronecker product structure.
    
    # M_{2,4,8} = M_{2,2,2} ⊗ M_{1,2,4}
    # M_{2,2,2}: i_B in [0,2), j_B in [0,2), k_B in [0,2)
    # M_{1,2,4}: i_A = 0, j_A in [0,2), k_A in [0,4)
    # Combined:
    #   i = i_B * 1 + i_A = i_B (since i_A=0)
    #   j = j_B * 2 + j_A (j_B in [0,2), j_A in [0,2)) -> j in [0,4)
    #   k = k_B * 4 + k_A (k_B in [0,2), k_A in [0,4)) -> k in [0,8)
    
    # So:
    #   U index: i * m + j = i_B * 4 + (j_B * 2 + j_A)
    #   V index: j * p + k = (j_B * 2 + j_A) * 8 + (k_B * 4 + k_A)
    #   W index: k * n + i = (k_B * 4 + k_A) * 2 + i_B
    
    # For each (r_B, r_A) with r_B in [0,7), r_A in [0,4):
    #   r = r_B * 4 + r_A
    #   U[i_B * 4 + j_B * 2 + j_A, r] += U_B[i_B * 2 + j_B, r_B] * U_A[j_A, r_A]
    #   V[(j_B * 2 + j_A) * 8 + k_B * 4 + k_A, r] += V_B[j_B * 2 + k_B, r_B] * V_A[j_A * 4 + k_A, r_A]
    #   W[(k_B * 4 + k_A) * 2 + i_B, r] += W_B[k_B * 2 + i_B, r_B] * W_A[k_A, r_A]
    
    for r_B in range(7):
        for r_A in range(4):
            r = r_B * 4 + r_A
            for i_B in range(2):
                for j_B in range(2):
                    for j_A in range(2):
                        u_val = U_B[i_B * 2 + j_B, r_B] * (1.0 if j_A == r_A else 0.0)
                        if u_val != 0:
                            U[i_B * 4 + j_B * 2 + j_A, r] += u_val
                        for k_B in range(2):
                            for k_A in range(4):
                                v_val = V_B[j_B * 2 + k_B, r_B] * (1.0 if j_A == r_A else 0.0)
                                if v_val != 0:
                                    V[(j_B * 2 + j_A) * 8 + k_B * 4 + k_A, r] += v_val
                                w_val = W_B[k_B * 2 + i_B, r_B] * 1.0
                                if w_val != 0:
                                    W[(k_B * 4 + k_A) * 2 + i_B, r] += w_val
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook (rank 64)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
