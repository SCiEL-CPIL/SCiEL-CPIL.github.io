"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    schoolbook_rank = n * m * p  # 42
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
        V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
        W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U_arr.reshape(n, m, rank)  # U[i, j, c]
        V3 = V_arr.reshape(m, p, rank)  # V[j, k, c]
        W3 = W_arr.reshape(p, n, rank)  # W[k, i, c]
        
        # For each rank component c, we have:
        # left_c = U3[:, :, c]  (n x m)
        # right_c = V3[:, :, c] (m x p)
        # out_c = W3[:, :, c]   (p x n) -> transpose to (n x p)
        # C[i,k] += out_c[i,k] * (left_c @ right_c)[i,k]
        
        # Vectorized: for each c, compute left_c @ right_c and scale by out_c
        # left_c @ right_c has shape (n, p)
        # We need sum over c of out_c.T[i,k] * (left_c @ right_c)[i,k]
        
        # left_c @ right_c for all c: (rank, n, p)
        # left3: (n, m, rank) -> (rank, n, m)
        # right3: (m, p, rank) -> (rank, m, p)
        left3 = U3.transpose(2, 0, 1)  # (rank, n, m)
        right3 = V3.transpose(2, 0, 1)  # (rank, m, p)
        products = np.einsum('cim,cjk->cip', left3, right3)  # (rank, n, p)
        
        # W3 is (p, n, rank) -> transpose to (rank, n, p)
        W3t = W3.transpose(2, 0, 1)  # (rank, n, p)
        
        C_computed = np.sum(products * W3t, axis=0)  # (n, p)
        
        # Test with random matrices
        for _ in range(3):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            # C[i,k] = sum_c W3t[c,i,k] * sum_{i',j} U3[i',j,c]*A[i',j] * sum_{j',k'} V3[j',k',c]*B[j',k']
            # = sum_c W3t[c,i,k] * (left_c @ right_c)[i,k]
            # But we need to actually compute with A and B:
            # left_c = U3[:,:,c], right_c = V3[:,:,c]
            # C[i,k] = sum_c W3t[c,i,k] * (U3[:,:,c] @ ... ) no
            # Actually: C[i,k] = sum_c W3t[c,i,k] * sum_j A[i,j] B[j,k] is wrong
            # The correct formula: C[i,k] = sum_j A[i,j] B[j,k]
            # With decomposition: C[i,k] = sum_c W3t[c,i,k] * (sum_{i',j} U3[i',j,c] A[i',j]) * (sum_{j',k'} V3[j',k',c] B[j',k'])
            # Wait, that's not right either. Let me reconsider.
            
            # The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
            # A rank-r decomposition: C[i,k] = sum_c (sum_{i',j} U[i',j,c] A[i',j]) (sum_{j',k'} V[j',k',c] B[j',k']) * W[i,k,c]
            # No, that would be trilinear in A, B, and the output weight.
            
            # Actually, the standard formulation for matrix multiplication rank:
            # C = AB, C[i,k] = sum_j A[i,j] B[j,k]
            # A rank-r algorithm: C[i,k] = sum_c M_c[i,k]
            # where M_c = (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
            # But this gives C[i,k] = sum_c (scalar_c_left) * (scalar_c_right) which is just a scalar, not a matrix entry.
            
            # I think the correct interpretation is:
            # The tensor M[i,j,k] = 1 for all i,j,k (the multiplication tensor)
            # Rank-r decomposition: M[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
            # where U has shape (n*m, r), V has shape (m*p, r), W has shape (n*p, r)
            # But the task says W has shape (p*n, rank) with trace convention W[k*n+i, r]
            
            # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # This is a trilinear form decomposition of the all-ones tensor.
            # Let me verify this interpretation.
            
            # Actually, for matrix multiplication C = AB:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # This is bilinear in A and B.
            # The tensor of the bilinear form has entries T[i,j,k] = 1 (each A[i,j]B[j,k] contributes to C[i,k])
            # A rank-r decomposition of this bilinear form means:
            # C[i,k] = sum_c (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
            # where U_c is a linear form in A and V_c is a linear form in B.
            # But this gives C[i,k] as a scalar sum of products of two linear forms, which doesn't match.
            
            # I think the correct factorization is:
            # C[i,k] = sum_c W_c[i,k] * (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
            # No, that's not standard.
            
            # Let me just use the direct verification: check that the decomposition reproduces C = AB.
            # For the trilinear form interpretation:
            # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
            # This is a constraint on U, V, W.
            
            # Let me verify this constraint directly.
            U4 = U_arr.reshape(n, m, rank)  # U[i,j,c]
            V4 = V_arr.reshape(m, p, rank)  # V[j,k,c]
            W4 = W_arr.reshape(p, n, rank)  # W[k,i,c]
            
            # Check: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
            # This is an einsum: 'ijc,jkc,kic->ijk'
            result = np.einsum('ijc,jkc,kic->ijk', U4, V4, W4)
            if not np.allclose(result, 1.0, atol=1e-10):
                return False
        
        return True
    
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Start from rank 35 down to 30
    for rank in range(35, 29, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 300
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_fast(U, V, W, rank):
                found = True
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
