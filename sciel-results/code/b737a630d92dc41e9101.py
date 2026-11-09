"""Matrix multiplication 3x3x3: rank-minimal exact tensor decomposition."""
import time
import sys
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
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

    # Try to find a lower-rank decomposition by searching for linear dependencies
    # among the schoolbook basis vectors. If the tensor has rank < 27, there exist
    # non-trivial linear combinations of the basis tensors that sum to zero.
    
    # For 3x3 matrix multiplication, the known rank is 23 (Strassen-like) or possibly lower.
    # Let's attempt a search for rank reduction by looking for linear dependencies
    # in the space of decompositions.
    
    # The matrix multiplication tensor T has components T[i,j,k] = delta[i,j,k] 
    # (where it's 1 if (i,j,k) corresponds to a valid product term).
    # Actually, for matrix multiplication, T[i,j,k] = 1 if the (i,j) entry of the first
    # matrix times the (j,k) entry of the second gives the (i,k) entry of the product.
    # So T[i,j,k] = delta_{i,k} is wrong. Let me re-derive.
    
    # The multiplication tensor has components:
    # T[i,j,k,l] = delta_{j,l} (the j-th column of A times k-th row of B contributes to (i,k) if j matches)
    # Wait, let me be more careful.
    
    # For matrix multiplication C = AB where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the structural tensor, all entries are 1 in the standard basis)
    # No, that's not right either.
    
    # The matrix multiplication tensor is a 3-tensor T in U^* x V^* x W^* where:
    # U has dimension n*m (entries of A), V has dimension m*p (entries of B), W has dimension p*n (entries of C)
    # T[u,v,w] = 1 if u corresponds to A[i,j], v corresponds to B[j,k], and w corresponds to C[i,k]
    # for the same (i,j,k). So T is a sum of 27 rank-1 tensors.
    
    # The question is whether these 27 rank-1 tensors are linearly independent.
    # If not, we can find a lower-rank decomposition.
    
    # Let's set up the system: we want to find coefficients c_1, ..., c_27 not all zero
    # such that sum_l c_l * e_{i_l,j_l}^U ⊗ e_{j_l,k_l}^V ⊗ e_{k_l,i_l}^W = 0
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in 27 unknowns.
    # For n=m=p=3, that's 9*9*9 = 729 equations in 27 unknowns.
    
    # Let's use numpy to solve this.
    
    try:
        import numpy as np
        
        # Build the matrix where each column corresponds to one schoolbook term
        # and each row corresponds to one entry in the full tensor space
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Total tensor space dimension
        total_dim = nm * mp * pn
        
        # Build the matrix: for each of the 27 terms, place a 1 at the appropriate position
        A_matrix = np.zeros((total_dim, rank), dtype=float)
        
        for l in range(rank):
            i = l // (m * p)
            jk = l % (m * p)
            j = jk // p
            k = jk % p
            
            # u index: i*m + j
            # v index: j*p + k  
            # w index: k*n + i
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Flatten index in the tensor space
            flat_idx = (u_idx * mp + v_idx) * pn + w_idx
            A_matrix[flat_idx, l] = 1.0
        
        # Find the null space of A_matrix
        # If the null space is non-trivial, we can reduce the rank
        
        # Use SVD to find the null space
        U_svd, s, Vt_svd = np.linalg.svd(A_matrix, full_matrices=True)
        
        # The number of nonzero singular values tells us the rank
        # Tolerance for considering a singular value as nonzero
        tol = 1e-10
        num_nonzero = sum(1 for sv in s if sv > tol)
        
        actual_rank = num_nonzero
        
        if actual_rank < rank:
            # We can reduce the rank!
            # The columns of A_matrix corresponding to the null space give linear dependencies
            # Let's extract a basis for the column space
            # The first actual_rank columns of Vt_svd.T span the row space
            # So we can use the first actual_rank columns of the identity... 
            # Actually, we need to find a basis for the column space of A_matrix
            
            # The right singular vectors corresponding to nonzero singular values span the column space
            # So the first actual_rank rows of Vt_svd give a basis for the column space
            # Wait, Vt_svd is (rank, rank) if full_matrices=False, or (rank, total_dim) if full_matrices=True
            
            # Let me re-do this more carefully
            U_svd, s, Vt_svd = np.linalg.svd(A_matrix, full_matrices=False)
            
            num_nonzero = sum(1 for sv in s if sv > tol)
            actual_rank = num_nonzero
            
            if actual_rank < rank:
                # The first actual_rank right singular vectors span the column space
                # Vt_svd has shape (rank, rank) with full_matrices=False
                # The first actual_rank rows of Vt_svd are the basis for the row space
                # The first actual_rank columns of U_svd are the basis for the column space
                
                # To get a lower-rank decomposition, we can use the projection
                # onto the column space
                # The new U, V, W will have actual_rank columns
                
                # Project the original U, V, W onto the column space basis
                # The basis is given by the first actual_rank columns of U_svd
                # But U_svd has shape (total_dim, rank), so the first actual_rank columns
                # are a basis for the column space of A_matrix
                
                # Actually, let me think about this differently.
                # If the 27 vectors (columns of A_matrix) are linearly dependent,
                # then there exists a matrix M of shape (rank, actual_rank) such that
                # A_matrix @ M = A_matrix_reduced (with actual_rank columns that span the same space)
                
                # The new decomposition will have:
                # U_new = U_old @ M
                # V_new = V_old @ M  
                # W_new = W_old @ M
                # where M is (rank, actual_rank)
                
                # We can use the QR decomposition or SVD to find M
                # The first actual_rank right singular vectors of A_matrix give a basis
                # for the column space. So M = Vt_svd[:actual_rank, :].T
                # Wait, Vt_svd[:actual_rank, :] gives actual_rank x rank matrix
                # Its transpose is rank x actual_rank
                # Let me verify: A_matrix @ (Vt_svd[:actual_rank, :].T) should give a full column rank matrix
                
                M = Vt_svd[:actual_rank, :].T  # shape (rank, actual_rank)
                
                # Convert U, V, W to numpy for easier manipulation
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                U_new = U_np @ M
                V_new = V_np @ M
                W_new = W_np @ M
                
                # Convert back to lists
                U_result = U_new.tolist()
                V_result = V_new.tolist()
                W_result = W_new.tolist()
                
                # Round small values to zero for cleanliness
                for mat in [U_result, V_result, W_result]:
                    for row in mat:
                        for idx, val in enumerate(row):
                            if abs(val) < 1e-9:
                                row[idx] = 0.0
                            elif abs(val - round(val)) < 1e-9:
                                row[idx] = float(round(val))
                            elif abs(val - round(val * 2) / 2) < 1e-9:
                                row[idx] = float(round(val * 2) / 2)
                
                return {"U": U_result, "V": V_result, "W": W_result}
        
    except ImportError:
        pass
    except Exception:
        pass

    # Fall back to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
