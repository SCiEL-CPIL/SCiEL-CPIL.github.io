"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    def tensor_residual(U, V, W):
        """Compute residual efficiently using matrix operations."""
        rank = U.shape[1]
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Compute the reconstructed tensor: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum for efficiency
        # 'ijr,jkr,kir->ijk'
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # The target tensor is all ones
        target = np.ones((n, m, p))
        
        # Return sum of squared errors
        return np.sum((reconstructed - target) ** 2)

    def is_half_integer(arr, tol=1e-6):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-5) and is_half_integer(V, 1e-5) and is_half_integer(W, 1e-5)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol

    # Constructive rank-10 solution:
    # Decompose T_{2x3x3} as direct sum of T_{2x2x2} (rank 7) and T_{1x3x3} (rank 3)
    # 
    # For the 2x2x2 part (indices i in {0,1}, j in {0,1}, k in {0,1}):
    # Use Strassen's algorithm with rank 7
    #
    # For the 1x3x3 part (indices i=0, j in {0,1,2}, k in {0,1,2}):
    # This is just a 1x3 vector times 3x3 matrix, which has rank 3
    #
    # Total rank = 7 + 3 = 10
    
    def strassen_2x2x2():
        """Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = A * B
        #
        # The 7 products are:
        # p1 = (a + d) * (e + h)
        # p2 = (c + d) * e
        # p3 = a * (f - h)
        # p4 = d * (g - e)
        # p5 = (a + b) * h
        # p6 = (c - a) * (e + f)
        # p7 = (b - d) * (g + h)
        #
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        #
        # In tensor decomposition form, for each product r, we have:
        # U[i,j,r], V[j,k,r], W[k,i,r] such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        
        rank = 7
        U = np.zeros((4, rank))  # 2*2 = 4 entries
        V = np.zeros((4, rank))  # 2*2 = 4 entries
        W = np.zeros((4, rank))  # 2*2 = 4 entries
        
        # Product 1: (a + d) * (e + h)
        # U[0,0] (a), U[1,1] (d) contribute to first factor
        # V[0,0] (e), V[1,1] (h) contribute to second factor
        # W: all ones for this product
        U[0, 0] = 1  # a
        U[1, 0] = 1  # d (index 1*2+1 = 3? No, i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3)
        # Wait, let me re-index: U[i*2+j, r]
        U[0, 0] = 1  # i=0,j=0 (a)
        U[3, 0] = 1  # i=1,j=1 (d)
        V[0, 0] = 1  # j=0,k=0 (e)
        V[3, 0] = 1  # j=1,k=1 (h)
        W[:, 0] = 1  # all W entries for this product
        
        # Product 2: (c + d) * e
        U[2, 1] = 1  # i=1,j=0 (c)
        U[3, 1] = 1  # i=1,j=1 (d)
        V[0, 1] = 1  # j=0,k=0 (e)
        V[1, 1] = 0  # j=0,k=1 (f) - not used
        V[2, 1] = 0  # j=1,k=0 (g) - not used
        V[3, 1] = 0  # j=1,k=1 (h) - not used
        W[:, 1] = 1
        
        # Product 3: a * (f - h)
        U[0, 2] = 1  # i=0,j=0 (a)
        V[1, 2] = 1  # j=0,k=1 (f)
        V[3, 2] = -1 # j=1,k=1 (h)
        W[:, 2] = 1
        
        # Product 4: d * (g - e)
        U[3, 3] = 1  # i=1,j=1 (d)
        V[2, 3] = 1  # j=1,k=0 (g)
        V[0, 3] = -1 # j=0,k=0 (e)
        W[:, 3] = 1
        
        # Product 5: (a + b) * h
        U[0, 4] = 1  # i=0,j=0 (a)
        U[1, 4] = 1  # i=0,j=1 (b)
        V[3, 4] = 1  # j=1,k=1 (h)
        W[:, 4] = 1
        
        # Product 6: (c - a) * (e + f)
        U[2, 5] = 1  # i=1,j=0 (c)
        U[0, 5] = -1 # i=0,j=0 (a)
        V[0, 5] = 1  # j=0,k=0 (e)
        V[1, 5] = 1  # j=0,k=1 (f)
        W[:, 5] = 1
        
        # Product 7: (b - d) * (g + h)
        U[1, 6] = 1  # i=0,j=1 (b)
        U[3, 6] = -1 # i=1,j=1 (d)
        V[2, 6] = 1  # j=1,k=0 (g)
        V[3, 6] = 1  # j=1,k=1 (h)
        W[:, 6] = 1
        
        return U, V, W, rank
    
    def one_by_three_by_three():
        """Return rank-3 decomposition for 1x3x3 matrix multiplication (vector-matrix)."""
        # This is just C[i,k] = sum_j A[i,j] * B[j,k] for i=0
        # So C[k] = sum_j A[j] * B[j,k]
        # This has rank 3, one for each j
        
        rank = 3
        U = np.zeros((3, rank))  # 1*3 = 3 entries (i=0, j=0,1,2)
        V = np.zeros((9, rank))  # 3*3 = 9 entries (j=0,1,2; k=0,1,2)
        W = np.zeros((3, rank))  # 3*1 = 3 entries (k=0,1,2; i=0)
        
        for j in range(3):
            U[j, j] = 1  # A[0,j]
            V[j*3 + 0, j] = 1  # B[j,0]
            V[j*3 + 1, j] = 1  # B[j,1]
            V[j*3 + 2, j] = 1  # B[j,2]
            W[0, j] = 1  # C[0,0]
            W[1, j] = 1  # C[0,1]
            W[2, j] = 1  # C[0,2]
        
        return U, V, W, rank
    
    def construct_rank_10():
        """Construct the rank-10 solution by combining Strassen and 1x3x3."""
        U_s, V_s, W_s, rank_s = strassen_2x2x2()
        U_v, V_v, W_v, rank_v = one_by_three_by_three()
        
        total_rank = rank_s + rank_v  # 7 + 3 = 10
        
        U = np.zeros((n * m, total_rank))  # 6 x 10
        V = np.zeros((m * p, total_rank))  # 9 x 10
        W = np.zeros((p * n, total_rank))  # 6 x 10
        
        # Copy Strassen part into first 7 columns
        # U_s is 4x7, corresponds to i in {0,1}, j in {0,1}
        # In U, indices are i*m + j = i*3 + j
        # So for i=0: j=0 -> 0, j=1 -> 1
        # For i=1: j=0 -> 3, j=1 -> 4
        U[0, :rank_s] = U_s[0, :]  # i=0, j=0
        U[1, :rank_s] = U_s[1, :]  # i=0, j=1
        U[3, :rank_s] = U_s[2, :]  # i=1, j=0
        U[4, :rank_s] = U_s[3, :]  # i=1, j=1
        
        # V_s is 4x7, corresponds to j in {0,1}, k in {0,1}
        # In V, indices are j*p + k = j*3 + k
        # For j=0: k=0 -> 0, k=1 -> 1
        # For j=1: k=0 -> 3, k=1 -> 4
        V[0, :rank_s] = V_s[0, :]  # j=0, k=0
        V[1, :rank_s] = V_s[1, :]  # j=0, k=1
        V[3, :rank_s] = V_s[2, :]  # j=1, k=0
        V[4, :rank_s] = V_s[3, :]  # j=1, k=1
        
        # W_s is 4x7, corresponds to k in {0,1}, i in {0,1}
        # In W, indices are k*n + i = k*2 + i
        # For k=0: i=0 -> 0, i=1 -> 1
        # For k=1: i=0 -> 2, i=1 -> 3
        W[0, :rank_s] = W_s[0, :]  # k=0, i=0
        W[1, :rank_s] = W_s[1, :]  # k=0, i=1
        W[2, :rank_s] = W_s[2, :]  # k=1, i=0
        W[3, :rank_s] = W_s[3, :]  # k=1, i=1
        
        # Copy 1x3x3 part into last 3 columns
        # U_v is 3x3, corresponds to i=0, j in {0,1,2}
        # In U, indices are i*m + j = 0*3 + j = j
        # So U[0], U[1], U[2] get the 1x3x3 part
        U[0, rank_s:] = U_v[0, :]  # i=0, j=0
        U[1, rank_s:] = U_v[1, :]  # i=0, j=1
        U[2, rank_s:] = U_v[2, :]  # i=0, j=2
        
        # V_v is 9x3, corresponds to j in {0,1,2}, k in {0,1,2}
        # In V, indices are j*p + k = j*3 + k
        # So V[0..2] for j=0, V[3..5] for j=1, V[6..8] for j=2
        V[:, rank_s:] = V_v  # All 9 entries
        
        # W_v is 3x3, corresponds to k in {0,1,2}, i=0
        # In W, indices are k*n + i = k*2 + 0 = k*2
        # So W[0], W[2], W[4] get the 1x3x3 part
        W[0, rank_s:] = W_v[0, :]  # k=0, i=0
        W[2, rank_s:] = W_v[1, :]  # k=1, i=0
        W[4, rank_s:] = W_v[2, :]  # k=2, i=0
        
        return U, V, W, total_rank
    
    # Try the constructive rank-10 solution first
    U_10, V_10, W_10, rank_10 = construct_rank_10()
    
    if verify_solution(U_10, V_10, W_10, 1e-6):
        return {"U": U_10.tolist(), "V": V_10.tolist(), "W": W_10.tolist()}
    
    # If rank-10 doesn't work, fall back to schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
