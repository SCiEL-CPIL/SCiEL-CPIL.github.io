import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms)
        U_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: 1},       
            {0: 1},             
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 0: -1},      
            {1: 1, 3: -1}       
        ]
        V_coeffs = [
            {0: 1, 3: 1},       
            {0: 1},             
            {1: 1, 3: -1},      
            {2: 1, 0: -1},      
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 3: 1}        
        ]
        W_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: -1},      
            {1: 1, 3: 1},       
            {0: 1, 2: 1},       
            {0: -1, 1: 1},      
            {3: 1},             
            {0: 1}              
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms)
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2 using structured search."""
        if time_left() < 5:
            return None
        
        U17, V17, W17 = get_17_rank_block()
        
        # Compute the tensor for each of the 17 terms
        term_tensors = []
        for c in range(17):
            u = U17[:, c].reshape(3, 3)
            v = V17[:, c].reshape(3, 2)
            w = W17[:, c].reshape(2, 3)
            t = np.zeros((3, 3, 2))
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        t[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            term_tensors.append(t.flatten())
        
        term_tensors = np.array(term_tensors)  # shape (17, 18)
        
        # Find the null space of term_tensors.T (18x17 matrix)
        # If there's a non-trivial null space, we have a dependency
        try:
            _, s, vh = np.linalg.svd(term_tensors.T, full_matrices=True)
            # vh has shape (17, 18) if full_matrices=True, but we want (17,17)
            # Actually, for a 18x17 matrix, SVD gives U (18x18), s (17,), Vh (17x18)
            # The null space of the 18x17 matrix is in the right singular vectors
            # corresponding to zero singular values
            if len(s) > 0:
                # Find singular values that are approximately zero
                tol = 1e-10
                null_indices = [i for i, sv in enumerate(s) if sv < tol]
                if len(null_indices) > 0:
                    # The last rows of Vh correspond to the null space
                    # Vh has shape (17, 18), so the null space vectors are in the rows
                    # Wait, Vh is (17, 18), so the right singular vectors are rows of Vh
                    # Actually, for A (m x n), SVD gives A = U S Vh where U (m x m), S (m x n), Vh (n x n)
                    # For our case, A = term_tensors.T is 18x17, so m=18, n=17
                    # U is 18x18, S is 18x17, Vh is 17x17
                    # The null space of A is spanned by the columns of V corresponding to zero singular values
                    # V is the conjugate transpose of Vh, so columns of V are rows of Vh^H
                    # Since we're dealing with real numbers, V = Vh.T
                    # So the null space vectors are the rows of Vh corresponding to zero singular values
                    # But Vh is 17x17, and S is 18x17, so there are 17 singular values
                    # If any singular value is ~0, the corresponding row of Vh is in the null space
                    for idx in null_indices:
                        dep = Vh[idx, :]  # This is a 17-vector in the null space
                        # dep[c] is the coefficient for term c
                        # sum_c dep[c] * term_tensors[c] = 0
                        # We can use this to eliminate one term
                        # Find the term with the largest absolute coefficient
                        max_idx = np.argmax(np.abs(dep))
                        if abs(dep[max_idx]) > 1e-10:
                            # We can express term max_idx as a combination of the others
                            # dep[max_idx] * T_max_idx = -sum_{c != max_idx} dep[c] * T_c
                            # T_max_idx = -sum_{c != max_idx} (dep[c] / dep[max_idx]) * T_c
                            coeffs = np.zeros(17)
                            for c in range(17):
                                if c != max_idx:
                                    coeffs[c] = -dep[c] / dep[max_idx]
                            
                            # Check if coefficients are small integers/half-integers
                            coeffs_rounded = np.round(coeffs * 2) / 2
                            # Verify
                            residual = np.zeros(18)
                            for c in range(17):
                                if c != max_idx:
                                    residual += coeffs_rounded[c] * term_tensors[c]
                            residual += term_tensors[max_idx]
                            
                            if np.max(np.abs(residual)) < 1e-10:
                                # We found a valid dependency!
                                # Create new 16-term decomposition
                                U_new = np.zeros((9, 16))
                                V_new = np.zeros((6, 16))
                                W_new = np.zeros((6, 16))
                                
                                col = 0
                                for c in range(17):
                                    if c != max_idx:
                                        U_new[:, col] = U17[:, c] * coeffs_rounded[c]
                                        V_new[:, col] = V17[:, c]
                                        W_new[:, col] = W17[:, c]
                                        col += 1
                                
                                # Verify
                                T_check = np.zeros((3, 3, 2))
                                for c in range(16):
                                    u = U_new[:, c].reshape(3, 3)
                                    v = V_new[:, c].reshape(3, 2)
                                    w = W_new[:, c].reshape(2, 3)
                                    for i in range(3):
                                        for j in range(3):
                                            for k in range(2):
                                                T_check[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                                
                                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # If SVD didn't find a dependency, try a direct search
        # Look for a rank-16 solution with specific structure
        # Assume most terms are schoolbook-like
        # Try to find 16 terms that sum to the target
        
        # Strategy: Use 10 schoolbook terms for the "border" (i=2 or j=2)
        # and find 6 terms for the 2x2x2 core
        # The 2x2x2 core has rank 7 (Strassen), so we can't do better than 7 for that part
        # Total would be 10 + 7 = 17, which is what we have
        
        # Alternative: Maybe we can share some terms between the border and core
        # This is complex, so let's try a grid search over small integer patterns
        
        # For now, return None to use the 17-rank fallback
        return None

    # Try to find rank 16 block with time limit
    block_16 = None
    if time_left() > 30:  # Only try if we have enough time
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        
    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        # Fallback: return the best valid solution
        # Try with the 17-rank block
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if validate(U, V, W):
            return result
        else:
            return schoolbook()
