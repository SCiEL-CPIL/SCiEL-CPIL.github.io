"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 40
    best_result = None
    best_rank = target_rank
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build the target tensor indices
    # T[idx_u, idx_v, idx_w] = 1 if idx_u=i*m+j, idx_v=j*p+k, idx_w=k*n+i for some i,j,k
    one_positions = []  # list of (idx_u, idx_v, idx_w) where T=1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                one_positions.append((idx_u, idx_v, idx_w))
    
    total_positions = n * m * m * p * p * n
    # Actually, the tensor has shape (n*m, m*p, p*n) = (8, 20, 10)
    # Total entries = 8 * 20 * 10 = 1600
    # Non-zero entries = 40 (one per (i,j,k))
    # Zero entries = 1560
    
    # For rank r decomposition:
    # U: (n*m, r) = (8, r)
    # V: (m*p, r) = (20, r)  
    # W: (p*n, r) = (10, r)
    # Variables: 8*r + 20*r + 10*r = 38*r
    # Constraints: 1600 (each entry must match)
    
    # For rank 39: 38*39 = 1482 variables, 1600 constraints
    # This is underdetermined, so a solution likely exists
    # We can set up a linear system
    
    def try_rank(r):
        """Try to find a decomposition of rank r using linear algebra."""
        if time.time() > deadline:
            return None
        
        # Number of variables
        num_u = n * m * r
        num_v = m * p * r
        num_w = p * n * r
        total_vars = num_u + num_v + num_w
        
        # Build the constraint matrix
        # For each (idx_u, idx_v, idx_w) in the tensor space:
        # sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c] = target[idx_u, idx_v, idx_w]
        # This is nonlinear! It's a product of three variables.
        # So we can't directly use linear algebra.
        
        # Alternative: fix one matrix and solve for the others?
        # Or use a different approach: since the tensor is sparse, 
        # we can use the fact that for rank r, we need r rank-1 terms.
        
        # Let's try a different approach: 
        # For each rank-1 term c, it contributes to positions where 
        # U[idx_u, c] * V[idx_v, c] * W[idx_w, c] != 0
        # We need the sum over c to give 1 at the 40 one-positions and 0 elsewhere.
        
        # This is still nonlinear. Let's try a heuristic search.
        
        # Approach: Start with schoolbook (rank 40) and try to merge/eliminate terms
        # Or: Try to find a solution where some entries are zero to reduce rank
        
        # For now, let's just return None and use schoolbook
        return None
    
    # Try to find lower rank decompositions
    # For this small case, let's try a direct approach:
    # The tensor T can be written as sum of rank-1 tensors
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # One approach: use the fact that for matrix multiplication,
    # there are known lower bounds and constructions
    
    # For 2×4×5, let's try to see if rank 39 works
    # by checking if we can find a non-trivial linear dependency
    
    # Actually, let's try a more practical approach:
    # Use the schoolbook decomposition and check if any column can be removed
    # by adjusting the others
    
    # Build schoolbook
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Check if we can reduce rank by finding a linear combination
    # that allows removing one term
    
    # For rank r-1, we need to find U', V', W' of rank r-1 such that
    # sum_{c=0}^{r-2} U'[:,c] V'[:,c]^T W'[:,c]^T = T
    
    # This is equivalent to finding a decomposition of T with r-1 terms.
    
    # Let's try a simple approach: check if the 40 rank-1 tensors are linearly independent
    # in the tensor space. If they are, rank is 40. If not, rank < 40.
    
    # Flatten each rank-1 tensor to a vector of length 8*20*10 = 1600
    # Then check if they are linearly independent
    
    def tensor_to_vector(u, v, w):
        """Convert a rank-1 tensor (u, v, w) to a flat vector."""
        vec = np.zeros((n * m, m * p, p * n))
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    vec[idx_u, idx_v, idx_w] = u[idx_u] * v[idx_v] * w[idx_w]
        return vec.flatten()
    
    # Build the 40 rank-1 tensors from schoolbook
    rank1_tensors = []
    for c in range(target_rank):
        u = U_school[:, c]
        v = V_school[:, c]
        w = W_school[:, c]
        vec = tensor_to_vector(u, v, w)
        rank1_tensors.append(vec)
    
    # Stack into a matrix and check rank
    tensor_matrix = np.array(rank1_tensors).T  # shape (1600, 40)
    
    # Compute the rank of this matrix
    # If rank < 40, then there's a linear dependency, and we might be able to reduce the decomposition rank
    
    try:
        matrix_rank = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
    except:
        matrix_rank = target_rank
    
    # If matrix_rank < 40, we have a linear dependency
    # But this doesn't directly give us a lower-rank decomposition of T
    # It just means the 40 rank-1 tensors are linearly dependent as vectors
    
    # For a lower-rank decomposition of T, we need to find r < 40 rank-1 tensors
    # that sum to T. This is different from the linear independence of the schoolbook terms.
    
    # Let's try a different approach: use the null space to find a dependency
    if matrix_rank < target_rank:
        # Find the null space
        try:
            Q, R = np.linalg.qr(tensor_matrix)
            # The last (40 - matrix_rank) columns of Q are in the null space
            null_vecs = Q[:, matrix_rank:]
            # For each null vector, we have sum_c null_vec[c] * tensor_c = 0
            # This means we can add a multiple of this to the decomposition
            # But we need to maintain that T = sum of rank-1 tensors
            
            # Actually, this tells us that the schoolbook terms are linearly dependent,
            # but it doesn't directly help us find a lower-rank decomposition of T.
            
            # Let's try to use this to construct a rank 39 decomposition
            # If there's a null vector a such that sum_c a[c] * (u_c ⊗ v_c ⊗ w_c) = 0,
            # then we can modify the decomposition by adding t * (a[c] * u_c ⊗ v_c ⊗ w_c)
            # But this doesn't reduce the rank unless we can make some coefficient zero.
            
            pass
        except:
            pass
    
    # For now, if we can't find a lower rank quickly, return schoolbook
    # Convert to lists
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
