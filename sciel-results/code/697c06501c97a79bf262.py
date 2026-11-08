import numpy as np
import time
import random
import itertools

TARGET_DEADLINE = time.time() + 110  # Safety margin

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition using NumPy for speed.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    
    # Check all non-zero entries of the target tensor
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check zero entries
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            for j_a in range(m):
                for j_b in range(m):
                    if j_a == j_b:
                        continue
                    a_idx = i * m + j_a
                    b_idx = j_b * p + k
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    if abs(val) > 1e-9:
                        return False
    return True

def _khatri_rao_solve(U, V, n, m, p, R):
    """
    Given U (nm x R) and V (mp x R), solve for W (pn x R).
    """
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                W[c_idx, :] = w
            except np.linalg.LinAlgError:
                return None
    return W

def _try_column_elimination(U, V, W, n, m, p, R):
    """
    Try to eliminate one column by checking if it's a linear combination of others.
    Returns (new_U, new_V, new_W, new_R) if successful, else None.
    """
    # For each column r, check if it can be expressed as a combination of others
    for r_elim in range(R):
        if time.time() > TARGET_DEADLINE - 15:
            return None
        
        # Build matrices for all other columns
        other_cols = [c for c in range(R) if c != r_elim]
        
        # For U: check if U[:, r_elim] is in span of U[:, other_cols]
        U_other = U[:, other_cols]
        U_target = U[:, r_elim]
        
        # Check if U_target is in column space of U_other
        U_full = np.column_stack([U_other, U_target.reshape(-1, 1)])
        rank_other = np.linalg.matrix_rank(U_other)
        rank_full = np.linalg.matrix_rank(U_full)
        
        if rank_full > rank_other:
            continue  # Not a linear combination
        
        # Similarly for V
        V_other = V[:, other_cols]
        V_target = V[:, r_elim]
        V_full = np.column_stack([V_other, V_target.reshape(-1, 1)])
        rank_other_v = np.linalg.matrix_rank(V_other)
        rank_full_v = np.linalg.matrix_rank(V_full)
        
        if rank_full_v > rank_other_v:
            continue
        
        # Similarly for W
        W_other = W[:, other_cols]
        W_target = W[:, r_elim]
        W_full = np.column_stack([W_other, W_target.reshape(-1, 1)])
        rank_other_w = np.linalg.matrix_rank(W_other)
        rank_full_w = np.linalg.matrix_rank(W_full)
        
        if rank_full_w > rank_other_w:
            continue
        
        # All three are linear combinations - try to eliminate this column
        # Solve for coefficients
        try:
            coeff_U, res_U, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
            coeff_V, res_V, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
            coeff_W, res_W, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            
            if res_U[0] > 1e-8 or res_V[0] > 1e-8 or res_W[0] > 1e-8:
                continue
            
            # Create new matrices without column r_elim
            new_U = np.column_stack([U[:, c] for c in other_cols])
            new_V = np.column_stack([V[:, c] for c in other_cols])
            new_W = np.column_stack([W[:, c] for c in other_cols])
            
            # Verify the new solution
            if _verify(new_U, new_V, new_W, n, m, p):
                return new_U, new_V, new_W, R - 1
        except np.linalg.LinAlgError:
            continue
    
    return None

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Trivial decomposition: Rank = n * m * p = 60
    R_trivial = n * m * p
    U_triv = np.zeros((nm, R_trivial))
    V_triv = np.zeros((mp, R_trivial))
    W_triv = np.zeros((pn, R_trivial))
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = (i * p + k) * m + j
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_triv[a_idx, r] = 1.0
                V_triv[b_idx, r] = 1.0
                W_triv[c_idx, r] = 1.0
                
    # Best solution found so far
    best_solution = {
        'U': U_triv.tolist(),
        'V': V_triv.tolist(),
        'W': W_triv.tolist()
    }
    best_rank = R_trivial
    
    rng = random.Random(42)
    
    # Try to find lower rank solutions
    for R in range(R_trivial - 1, 49, -1):
        if time.time() > TARGET_DEADLINE - 20:
            break
        
        max_iters = 500
        found = False
        
        for it in range(max_iters):
            if time.time() > TARGET_DEADLINE - 10:
                break
            
            # Structured random initialization
            density = 0.15 + 0.1 * (it % 4)
            vals = [-1.0, 0.5, 1.0, -0.5, 0.25, -0.25]
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(nm):
                for c in range(R):
                    if rng.random() < density:
                        U[r, c] = rng.choice(vals)
                        
            for r in range(mp):
                for c in range(R):
                    if rng.random() < density:
                        V[r, c] = rng.choice(vals)
            
            W = _khatri_rao_solve(U, V, n, m, p, R)
            
            if W is not None:
                if _verify(U, V, W, n, m, p):
                    # Try to eliminate columns
                    improved = True
                    while improved and time.time() < TARGET_DEADLINE - 5:
                        improved = False
                        elim_result = _try_column_elimination(U, V, W, n, m, p, U.shape[1])
                        if elim_result is not None:
                            U, V, W, R = elim_result
                            improved = True
                    
                    if R < best_rank:
                        best_rank = R
                        best_solution = {
                            'U': U.tolist(),
                            'V': V.tolist(),
                            'W': W.tolist()
                        }
                    found = True
                    break  # Found a solution for this rank, move to next
    
    return best_solution
