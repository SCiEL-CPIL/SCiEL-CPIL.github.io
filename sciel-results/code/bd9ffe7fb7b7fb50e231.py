import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen-based blocks plus aggressive rank-reduction search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def add_strassen_2x2x2(terms, i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block."""
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))
    
    def add_schoolbook_term(terms, i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    def build_initial_terms():
        """Build initial Strassen-based decomposition terms."""
        terms = []
        # 2x2x2 Strassen blocks for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        for i_start in [0]:
            for j_start in [0, 2]:
                for k_start in [0, 2]:
                    add_strassen_2x2x2(terms, i_start, j_start, k_start)
        
        # Remaining terms: j in {0,1,2,3}, k=4
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(terms, i, j, 4)
        
        # j=4, k in {0,1,2,3,4}
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(terms, i, 4, k)
        
        return terms
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W
    
    def try_reduce_rank_linear(U, V, W, max_time=120):
        """
        Attempt to reduce rank by removing terms and solving for one factor matrix.
        Uses multiple strategies: single removal, then repeated passes.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        def remove_term(U, V, W, r, fixed_factors):
            """
            Try to remove term r.
            fixed_factors: set of which factors are fixed (e.g., {'V', 'W'} means solve for U).
            """
            R = U.shape[1]
            if R <= 1:
                return None
            
            if 'U' in fixed_factors or 'V' in fixed_factors or 'W' in fixed_factors:
                # Determine which factor to solve for
                if 'V' in fixed_factors and 'W' in fixed_factors:
                    solve_for = 'U'
                elif 'U' in fixed_factors and 'W' in fixed_factors:
                    solve_for = 'V'
                elif 'U' in fixed_factors and 'V' in fixed_factors:
                    solve_for = 'W'
                else:
                    return None
            
            # Create indices for remaining terms
            idx = [i for i in range(R) if i != r]
            
            if solve_for == 'U':
                V_rem = V[:, idx]
                W_rem = W[:, idx]
                
                M = np.zeros((n * m, (m * p) * (p * n), len(idx)))
                T_flat = T.reshape(n * m, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(V[:, k_orig], W[:, k_orig])
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                U_new = np.zeros((n * m, R - 1))
                for a in range(n * m):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[a], T_flat[a], rcond=None)
                        if np.allclose(M[a] @ sol, T_flat[a], atol=1e-9):
                            U_new[a] = sol
                        else:
                            return None
                    except:
                        return None
                
                return U_new, V_rem, W_rem

            elif solve_for == 'V':
                U_rem = U[:, idx]
                W_rem = W[:, idx]
                
                M = np.zeros((m * p, (n * m) * (p * n), len(idx)))
                T_trans = np.transpose(T, (1, 0, 2))
                T_flat_b = T_trans.reshape(m * p, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(U[:, k_orig], W[:, k_orig])
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                V_new = np.zeros((m * p, R - 1))
                for b in range(m * p):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[b], T_flat_b[b], rcond=None)
                        if np.allclose(M[b] @ sol, T_flat_b[b], atol=1e-9):
                            V_new[b] = sol
                        else:
                            return None
                    except:
                        return None
                
                return U_rem, V_new, W_rem

            elif solve_for == 'W':
                U_rem = U[:, idx]
                V_rem = V[:, idx]
                
                M = np.zeros((p * n, (n * m) * (m * p), len(idx)))
                T_trans = np.transpose(T, (2, 0, 1))
                T_flat_c = T_trans.reshape(p * n, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(U[:, k_orig], V[:, k_orig])
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                W_new = np.zeros((p * n, R - 1))
                for c in range(p * n):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[c], T_flat_c[c], rcond=None)
                        if np.allclose(M[c] @ sol, T_flat_c[c], atol=1e-9):
                            W_new[c] = sol
                        else:
                            return None
                    except:
                        return None
                
                return U_rem, V_rem, W_new

        # Main loop: try all removals in all orders
        current_U, current_V, current_W = U, V, W
        progress = True
        while progress and time.time() < max_time:
            progress = False
            R = current_U.shape[1]
            for r in range(R):
                for fixed in [{'V', 'W'}, {'U', 'W'}, {'U', 'V'}]:
                    res = remove_term(current_U, current_V, current_W, r, fixed)
                    if res is not None:
                        current_U, current_V, current_W = res
                        progress = True
                        break
                if progress:
                    break
        
        return current_U, current_V, current_W, current_U.shape[1]
    
    def try_remove_multiple_terms(U, V, W, max_remove=3, max_time=60):
        """
        Try to remove multiple terms simultaneously using linear algebra.
        For a set S of terms to remove, check if the remaining terms can span T.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        best_U, best_V, best_W = U, V, W
        best_rank = R
        
        # Try removing 1 term (already handled by try_reduce_rank_linear)
        # Try removing 2 terms simultaneously
        for r1 in range(R):
            for r2 in range(r1 + 1, R):
                if time.time() > max_time:
                    break
                idx = [i for i in range(R) if i != r1 and i != r2]
                if len(idx) < 1:
                    continue
                
                # Check if we can solve for U with these removed
                V_rem = V[:, idx]
                W_rem = W[:, idx]
                
                M = np.zeros((n * m, (m * p) * (p * n), len(idx)))
                T_flat = T.reshape(n * m, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(V[:, k_orig], W[:, k_orig])
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                U_new = np.zeros((n * m, len(idx)))
                valid = True
                for a in range(n * m):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[a], T_flat[a], rcond=None)
                        if not np.allclose(M[a] @ sol, T_flat[a], atol=1e-9):
                            valid = False
                            break
                        U_new[a] = sol
                    except:
                        valid = False
                        break
                
                if valid:
                    best_U, best_V, best_W = U_new, V_rem, W_rem
                    best_rank = len(idx)
                    break
            
            if time.time() > max_time:
                break
        
        return best_U, best_V, best_W, best_rank
    
    # Build initial decomposition
    terms = build_initial_terms()
    U, V, W = build_decomposition_from_terms(terms)
    R_initial = U.shape[1]
    
    # Verify initial decomposition
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        R = n * m * p
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank - first single-term removals
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        U, V, W, R_reduced = try_reduce_rank_linear(U, V, W, max_time=min(remaining_time - 20, 80))
        
        # Verify reduced decomposition
        valid, err = verify_decomposition(U, V, W)
        if valid:
            R = R_reduced
        else:
            R = R_initial
    
    # Try multi-term removal if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 15 and U.shape[1] > 30:
        U2, V2, W2, R_multi = try_remove_multiple_terms(U, V, W, max_time=min(remaining_time - 5, 40))
        valid2, err2 = verify_decomposition(U2, V2, W2)
        if valid2 and R_multi < U.shape[1]:
            U, V, W = U2, V2, W2
            R = R_multi
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
