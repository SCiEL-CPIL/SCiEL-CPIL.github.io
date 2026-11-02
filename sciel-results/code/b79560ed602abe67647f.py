import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition (rank 72 for 4x4x5)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 = A00 B00 + A01 B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    # C01 = A00 B01 + A01 B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    # C10 = A10 B00 + A11 B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    # C11 = A10 B01 + A11 B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _solve_for_matrix(fixed1, fixed2, target_tensor, n, m, p, rank, var_index):
    """
    Solve for the variable matrix (U, V, or W) given the other two fixed.
    var_index: 0 for U, 1 for V, 2 for W.
    """
    U = fixed1 if var_index != 0 else None
    V = fixed2 if var_index != 1 else None
    W = fixed1 if var_index == 2 else (fixed2 if var_index == 2 else None)
    
    # We need to build a linear system A x = b
    # where x is the flattened variable matrix.
    
    # Target tensor T is of shape (n, m, p)
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # If solving for U (var_index=0):
    # U[i*m+j, r] is the unknown.
    # For each (i, j), we have a system for U[i*m+j, :].
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k.
    
    if var_index == 0: # Solve for U
        U_sol = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U_sol[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    return None
        return U_sol

    elif var_index == 1: # Solve for V
        V_sol = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                # A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V_sol[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    return None
        return V_sol

    elif var_index == 2: # Solve for W
        W_sol = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(p):
                # A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W_sol[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    return None
        return W_sol

def _try_reduce_rank(U0, V0, W0, n, m, p, target_rank, max_iters=50):
    """
    Try to find a decomposition of rank `target_rank` starting from rank 72 solution.
    Uses cyclic least squares.
    """
    # Initialize with first `target_rank` columns of the base solution
    U = U0[:, :target_rank].copy()
    V = V0[:, :target_rank].copy()
    W = W0[:, :target_rank].copy()
    
    # Add some noise to escape local minima? Or just try as is.
    # Let's try a few random permutations of columns to see if a different subset works.
    
    for _ in range(max_iters):
        # Solve for U
        new_U = _solve_for_matrix(V, W, None, n, m, p, target_rank, 0)
        if new_U is None: return None
        U = new_U
        
        # Solve for V
        new_V = _solve_for_matrix(U, W, None, n, m, p, target_rank, 1)
        if new_V is None: return None
        V = new_V
        
        # Solve for W
        new_W = _solve_for_matrix(U, V, None, n, m, p, target_rank, 2)
        if new_W is None: return None
        W = new_W
        
        # Round to half-integers
        U_r = _round_matrix_to_half_int(U)
        V_r = _round_matrix_to_half_int(V)
        W_r = _round_matrix_to_half_int(W)
        
        # Verify
        if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r, V_r, W_r
            
        # If not verified, continue iterating? 
        # The rounding might have pushed it away. 
        # We can use the rounded values as the new starting point for the next iteration?
        U = U_r
        V = V_r
        W = W_r
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get the Strassen-based decomposition as baseline (rank 72)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = U_base.shape[1]
    
    # Try to find lower-rank decompositions
    # We know rank 72 works. Try 71, 70, ...
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple random initializations (permutations of columns)
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed)
            # Permute columns of base solution to try different bases
            perm = np.random.permutation(best_rank)
            U_init = U_base[:, perm[:rank]]
            V_init = V_base[:, perm[:rank]]
            W_init = W_base[:, perm[:rank]]
            
            result = _try_reduce_rank(U_init, V_init, W_init, n, m, p, rank, max_iters=10)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
            
    return best_result
# EVOLVE-BLOCK-END
