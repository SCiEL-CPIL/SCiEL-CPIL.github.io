import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor for the full problem
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)

    # Fallback: Schoolbook solution
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    deadline = time.time() + 140 # Safety margin

    # Strategy: Find a rank-16 decomposition for 2x3x3 and use it 3 times to get rank 48.
    # Sub-problem: n_sub=2, m_sub=3, p_sub=3
    n_sub, m_sub, p_sub = 2, 3, 3
    nm_sub, mp_sub, pn_sub = n_sub * m_sub, m_sub * p_sub, p_sub * n_sub
    
    T_sub = np.zeros((nm_sub, mp_sub, pn_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.0

    def find_sub_decomposition(target_rank, deadline_inner):
        """Search for a rank-target_rank decomposition for 2x3x3."""
        # Try random V and W, solve for U
        for trial in range(2000):
            if time.time() > deadline_inner:
                break
            
            np.random.seed(42 + trial)
            # Sparse random matrices often work better for these small tensor ranks
            V = np.random.randint(-1, 2, size=(mp_sub, target_rank))
            W = np.random.randint(-1, 2, size=(pn_sub, target_rank))
            
            # Add some zeros to encourage sparsity if needed, but random int is fine
            # M matrix for solving U: M[b, r] = V[b, r] * W[c, r] ... no, it's a 3-way tensor.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed a, T[a, :, :] = sum_r U[a, r] (V[:, r] * W[:, r])
            # Let Z_r = V[:, r] * W[:, r] (element-wise product, shape mp_sub x pn_sub)
            # Then T[a, :, :] = sum_r U[a, r] Z_r
            # Flatten T[a, :, :] to vector t_a of length mp_sub * pn_sub
            # Flatten Z_r to vector z_r
            # t_a = U[a, :] @ Z_matrix where Z_matrix is (rank, mp_sub * pn_sub)
            
            Z_matrix = np.zeros((target_rank, mp_sub * pn_sub))
            for r in range(target_rank):
                Z_matrix[r, :] = (V[:, r] * W[:, r]).flatten()
            
            # Check rank of Z_matrix
            if np.linalg.matrix_rank(Z_matrix) < target_rank:
                continue
                
            U_candidate = np.zeros((nm_sub, target_rank))
            valid = True
            for a in range(nm_sub):
                t_a = T_sub[a, :, :].flatten()
                try:
                    # Solve Z_matrix.T @ U_candidate[a, :] = t_a
                    # This is an overdetermined system if we view it as Z_matrix.T x U = t
                    # Actually, t_a = sum_r U[a, r] Z_matrix[r, :]
                    # So t_a is a linear combination of rows of Z_matrix.
                    # We want to find coefficients U[a, :].
                    # This is a least squares problem: Z_matrix.T @ U_col = t_a
                    sol, residuals, rank_z, sv = np.linalg.lstsq(Z_matrix.T, t_a, rcond=None)
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                    U_candidate[a, :] = sol
                except:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            if is_half_integer(U_candidate):
                U_snapped = np.round(2.0 * U_candidate) / 2.0
                if verify_solution(U_snapped, V, W, T_sub):
                    return U_snapped, V, W
            else:
                # Try snapping anyway if close
                U_snapped = np.round(2.0 * U_candidate) / 2.0
                if verify_solution(U_snapped, V, W, T_sub):
                    return U_snapped, V, W

        return None

    # Try to find rank 16 decomposition for sub-problem
    sub_decomp = find_sub_decomposition(16, deadline - 10)
    
    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        # Construct full solution
        R_full = 3 * 16
        U_full = np.zeros((nm, R_full))
        V_full = np.zeros((mp, R_full))
        W_full = np.zeros((pn, R_full))
        
        for b in range(3):
            for r in range(16):
                col = b * 16 + r
                # U is the same for all blocks
                U_full[:, col] = U_sub[:, r]
                
                # V mapping: V_sub[j, k'] -> V_full[j*p + (3b + k'), col]
                for j in range(m_sub):
                    for k_prime in range(p_sub):
                        k_full = 3 * b + k_prime
                        V_full[j * p + k_full, col] = V_sub[j * p_sub + k_prime, r]
                        
                # W mapping: W_sub[k', i] -> W_full[(3b + k')*n + i, col]
                for k_prime in range(p_sub):
                    for i in range(n_sub):
                        k_full = 3 * b + k_prime
                        W_full[k_full * n + i, col] = W_sub[k_prime * n_sub + i, r]
                        
        if verify_solution(U_full, V_full, W_full, T_full):
            best_rank = R_full
            best_U, best_V, best_W = U_full, V_full, W_full
            # We could try to search for lower rank now, but 48 is a solid jump.
            # Let's check if we have time for a quick search for rank < 48 on the full problem?
            # Probably not worth the risk of complexity. 48 is a known theoretical improvement.
            
    # If sub_decomp failed, we stick with schoolbook 54.
    # We could try a direct search for rank 48 on the full problem, but it's harder.
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
