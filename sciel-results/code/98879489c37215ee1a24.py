import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2 rows of A, m=3 cols of A / rows of B, p=4 cols of B
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Try to find a decomposition with rank < 22
    # The tensor T has entries T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # In the trace convention: W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] should sum to delta for the multiplication
    
    # Actually, the tensor for matrix multiplication of n x m by m x p matrices has:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # The multiplication tensor is: sum over j of A[i,j] * B[j,k] = C[i,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k) - wait no.
    
    # The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is indexed by (i, j, k) and T[i,j,k] = 1 if this term appears
    # Actually for the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k in range
    # because every A[i,j] * B[j,k] contributes to C[i,k].
    
    # Wait, let me reconsider. The tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n} (with trace convention)
    # T_{(i,j),(j,k),(k,i)} = 1 if the product A_{i,j} B_{j,k} contributes to C_{i,k}
    # With trace convention W[k*n+i, r], the output index is k*n+i where k is column of C, i is row of C.
    # U[i*m+j, r] where i is row of A, j is column of A
    # V[j*p+k, r] where j is row of B, k is column of B
    
    # So T[(i,j), (j,k), (k,i)] = 1 for all valid indices
    # This is the matrix multiplication tensor.
    
    # For n=2, m=3, p=4:
    # U has 6 rows (i*m+j for i in 0..1, j in 0..2)
    # V has 12 rows (j*p+k for j in 0..2, k in 0..3)
    # W has 8 rows (k*n+i for k in 0..3, i in 0..1)
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,4)
    # Total number of non-zero entries = 2*3*4 = 24
    
    # We need to find rank-minimal decomposition.
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Also rank >= n*m*p / max(n,m,p) ... various bounds
    
    # Let's try a systematic approach: start with rank 12 and try to increase until we find a valid decomposition.
    # But that's expensive. Let's try known constructions first.
    
    # For 2x3x4, one approach:
    # Split into blocks. A is 2x3, B is 3x4.
    # We can think of it as: C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:3] @ B[2:3, 0:2]
    # and C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:3] @ B[2:3, 2:4]
    
    # The 2x2x2 part can use Strassen (rank 7), and the cross terms are rank 1 each.
    # C[:, 0:2]: 2x2x2 Strassen (7) + 2 rank-1 terms (for A[:,2] @ B[2,0:2]) = 9
    # C[:, 2:4]: 2x2x2 Strassen (7) + 2 rank-1 terms (for A[:,2] @ B[2,2:4]) = 9
    # Total: 18? But we can share some structure.
    
    # Actually, let me try a different approach. Let's use the fact that we can combine terms.
    
    # Let me try rank 16, 17, 18, 19, 20, 21 and see if any work.
    # I'll use a greedy/iterative approach to build the decomposition.
    
    # First, let's define the target tensor.
    # T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # In the flattened form:
    # U_index = i*m + j
    # V_index = j*p + k  
    # W_index = k*n + i
    
    # For a rank-r decomposition:
    # For each (i,j,k): sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to find a decomposition with rank r for r from 12 upward.
    
    def try_decomposition(r):
        """Try to find a rank-r decomposition using iterative refinement."""
        # Initialize with random or structured values
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Iteratively update each factor
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for idx in range(n * m):
                i = idx // m
                j = idx % m
                # We need: sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # For fixed (i,j), we need to find U[idx, :] such that
                # sum_r U[idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k in range(p)
                
                # This is a linear system: for each k, sum_r U[idx, r] * B_k[r] = 1
                # where B_k[r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Collect the equations
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve A_mat @ U[idx, :] = ones(p)
                # Use least squares
                if r >= p:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                else:
                    # Underdetermined, use pseudoinverse
                    sol = np.linalg.pinv(A_mat) @ np.ones(p)
                
                U[idx, :] = sol
            
            # Update V
            for idx in range(m * p):
                j = idx // p
                k = idx % p
                # We need: sum_r U[i*m+j, r] * V[idx, r] * W[k*n+i, r] = 1 for all i
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                if r >= n:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                else:
                    sol = np.linalg.pinv(A_mat) @ np.ones(n)
                
                V[idx, :] = sol
            
            # Update W
            for idx in range(p * n):
                k = idx // n
                i = idx % n
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[idx, r] = 1 for all j
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                if r >= m:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                else:
                    sol = np.linalg.pinv(A_mat) @ np.ones(m)
                
                W[idx, :] = sol
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += (val - 1) ** 2
            
            if error < 1e-8:
                # Found a valid decomposition! Now try to clean up to get integer/half-integer entries
                # For now, just return the solution
                return U, V, W
        
        return None
    
    # Try ranks from 12 upward
    best_result = None
    for r in range(12, 22):
        if time.time() > deadline - 10:
            break
        
        result = try_decomposition(r)
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find anything better, fall back to the known rank-22 solution
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)  # M1
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)  # M2
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)  # M3
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)  # M4
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)  # M5
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)  # M6
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)  # M7
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)  # M1'
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)  # M2'
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)  # M3'
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)  # M4'
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)  # M5'
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)  # M6'
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)  # M7'
    
    # Outer: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}
