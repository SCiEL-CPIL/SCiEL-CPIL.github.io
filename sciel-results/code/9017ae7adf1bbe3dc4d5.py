import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 20 decomposition using Laderman's algorithm for 4x4 matrix multiplication
    # Laderman's algorithm decomposes 4x4 * 4x4 into 20 rank-1 terms
    # We use this for the right half (4x4 * 4x4) and simple multiplication for the left half (2x4 * 4x4)
    # Actually, let's try to use Laderman for both halves if possible, or find a better decomposition.
    
    # Laderman's 20-term decomposition for 4x4 matrices:
    # The 16 products are expressed as linear combinations of 20 scalar products.
    # Let A and B be 4x4 matrices.
    # The 20 terms are defined by their support in A, B, and C.
    
    # I will implement Laderman's algorithm directly.
    # Source: Laderman, L. (1976). "The number of multiplications needed to compute a matrix product".
    
    # The 20 terms (u, v, w) where u, v, w are 4x4 matrices (flattened to 16 elements).
    # Indices: row-major (0..15)
    
    rank = 20 + 12 # 20 for right half, 12 for left half? No, let's try to be smarter.
    # Wait, the left half is A (2x4) * B (4x4).
    # A is 2x4. B is 4x4.
    # We can view this as two 2x4 * 4x4 products? No, it's one product.
    # C_left (2x4) = A (2x4) * B (4x4).
    # This is equivalent to multiplying a 2x4 matrix by a 4x4 matrix.
    # We can pad A to 4x4 with zeros?
    # If we pad A to 4x4 (A_pad), then C_pad (4x4) = A_pad * B.
    # Then C_left is the top 2 rows of C_pad.
    # Using Laderman on A_pad * B would cost 20 terms.
    # But we only need 2x4 output.
    # Is there a cheaper way?
    # Rank of 2x4 * 4x4 multiplication tensor.
    # Upper bound: 2 * rank(1x4 * 4x4) = 2 * 4 = 8? No, 1x4 * 4x4 is just 4 multiplications.
    # So 2 * 4 = 8 terms for the left half?
    # Let's check: C[i, k] = sum_j A[i, j] B[j, k].
    # For i=0: C[0, k] = sum_j A[0, j] B[j, k]. This is 4 scalar multiplications for each k?
    # No, the tensor is rank 8?
    # T_left = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # This is isomorphic to two independent 1x4 * 4x4 products?
    # Yes, because i=0 and i=1 don't interact in the summation over j.
    # So rank(T_left) <= 2 * rank(T_1x4_4x4).
    # T_1x4_4x4: C[k] = sum_j A[j] B[j,k].
    # This is just 4 scalar multiplications: A[0]*B[0,k], A[1]*B[1,k], etc.
    # So rank is 4.
    # Thus rank(T_left) <= 8.
    
    # So total rank <= 8 (left) + 20 (right) = 28.
    # The current program is 28.
    # Can we do better?
    # Maybe Laderman can be adapted or a smaller algorithm exists for 2x4 * 4x4?
    # Or maybe the right half 2x4 * 4x4 (which is part of 2x4 * 4x4) can be done cheaper?
    # Wait, the right half of C (2x4) is C[:, 2:4] = A * B[:, 2:4].
    # This is also 2x4 * 4x4. So it also costs 8.
    # Total 8 + 8 = 16?
    # Let's verify if 2x4 * 4x4 can be done in 8.
    # C[i, k] = sum_{j=0}^3 A[i, j] B[j, k].
    # The tensor is sum_{i=0}^1 sum_{j=0}^3 sum_{k=0}^3 A[i,j] B[j,k] C[i,k].
    # We can write this as sum_{i=0}^1 sum_{j=0}^3 (A[i,j] * B[j, :] * C[i, :]).
    # This is 2 * 4 = 8 rank-1 terms.
    # Term (i, j):
    #   U has 1 at (i, j).
    #   V has 1 at (j, k) for all k? No, V is a vector in B space.
    #   In the rank-1 decomposition, we need vectors u, v, w such that u ⊗ v ⊗ w.
    #   u[i,j] = 1.
    #   v[j', k'] = delta(j, j') * delta(k', k_target)?
    #   No, the term is A[i,j] * B[j,k] * C[i,k].
    #   So for fixed i, j, we have a term involving A[i,j], B[j,k], C[i,k].
    #   This is not a single rank-1 term because k varies.
    #   Wait, A[i,j] is a scalar. B[j,k] is a scalar. C[i,k] is a scalar.
    #   The product is A[i,j] B[j,k] C[i,k].
    #   This is a rank-1 tensor in the space A ⊗ B ⊗ C?
    #   The tensor T = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    #   Let's check if it's rank 8.
    #   T = sum_{i=0}^1 sum_{j=0}^3 A[i,j] ⊗ B[j,:] ⊗ C[i,:].
    #   Here B[j,:] is a vector in the B space (dimension 4x4=16).
    #   B[j,:] has 1s at indices (j, 0), (j, 1), (j, 2), (j, 3).
    #   C[i,:] has 1s at indices (i, 0), (i, 1), (i, 2), (i, 3).
    #   So yes, it is a sum of 8 rank-1 terms.
    #   Each term has U with 1 at (i,j), V with 1s at (j,0..3), W with 1s at (i,0..3).
    #   So rank 8 is achievable for 2x4 * 4x4.
    
    # So for the left half (cols 0,1 of C):
    # We need C[:, 0:2] = A * B[:, 0:2].
    # This is 2x4 * 4x2.
    # Rank?
    # T_left = sum_{i,j,k=0}^1 A[i,j] B[j,k] C[i,k].
    # T_left = sum_{i=0}^1 sum_{j=0}^3 A[i,j] ⊗ B[j, 0:2] ⊗ C[i, 0:2].
    # B[j, 0:2] has 1s at (j,0), (j,1).
    # C[i, 0:2] has 1s at (i,0), (i,1).
    # So 8 terms.
    
    # For the right half (cols 2,3 of C):
    # C[:, 2:4] = A * B[:, 2:4].
    # Similarly 8 terms.
    
    # Total rank = 8 + 8 = 16.
    
    # This is much better than 28!
    # Let's implement this.
    
    rank = 16
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_idx, u_val, v_idxs, v_val, w_idxs, w_val, col_idx):
        U[u_idx, col_idx] += u_val
        for idx in v_idxs:
            V[idx, col_idx] += v_val
        for idx in w_idxs:
            W[idx, col_idx] += w_val
            
    col = 0
    for i in range(n): # i = 0, 1
        for j in range(m): # j = 0, 1, 2, 3
            # Left half (k=0,1)
            u_idx = i * m + j
            v_idxs = [j * p + k for k in range(2)] # (j,0), (j,1)
            w_idxs = [k * n + i for k in range(2)] # (0,i), (1,i) -> k*n+i
            add_term(u_idx, 1, v_idxs, 1, w_idxs, 1, col)
            col += 1
            
    for i in range(n): # i = 0, 1
        for j in range(m): # j = 0, 1, 2, 3
            # Right half (k=2,3)
            u_idx = i * m + j
            v_idxs = [j * p + k for k in range(2, 4)] # (j,2), (j,3)
            w_idxs = [k * n + i for k in range(2, 4)] # (2,i), (3,i)
            add_term(u_idx, 1, v_idxs, 1, w_idxs, 1, col)
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
