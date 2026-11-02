import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_28 = 28
    
    # Time limit guard
    start_time = time.time()
    time_limit = 100.0 # seconds
    
    # Try to find rank 27
    rank_target = 27
    success = False
    U_sol = None
    V_sol = None
    W_sol = None
    
    # We will use a simple gradient descent with restarts
    # Number of restarts depends on time
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        np.random.seed(42 + restart) # Deterministic for reproducibility, but varied per restart
        U = np.random.randn(n * m, rank_target) * 0.1
        V = np.random.randn(m * p, rank_target) * 0.1
        W = np.random.randn(p * n, rank_target) * 0.1
        
        # Reshape for einsum
        # U: (n, m, R) -> (2, 4, 27)
        # V: (m, p, R) -> (4, 4, 27)
        # W: (p, n, R) -> (4, 2, 27)
        U_r = U.reshape(n, m, rank_target)
        V_r = V.reshape(m, p, rank_target)
        W_r = W.reshape(p, n, rank_target)
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U_r)
        vV = np.zeros_like(V_r)
        vW = np.zeros_like(W_r)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute F
            # F_ijk = sum_r U_ijr V_jkr W_kir
            F = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
            F_err = F - 1.0
            loss = np.sum(F_err ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                
            if loss < 1e-8:
                break
            
            # Gradients
            gU = 2.0 * np.einsum('ijk,jkr,kir->ijr', F_err, V_r, W_r, optimize=True)
            gV = 2.0 * np.einsum('ijk,ijr,kir->jkr', F_err, U_r, W_r, optimize=True)
            gW = 2.0 * np.einsum('ijk,ijr,jkr->kir', F_err, U_r, V_r, optimize=True)
            
            # Adam update (simplified)
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            U_r -= lr * vU
            V_r -= lr * vV
            W_r -= lr * vW
            
            # Decay learning rate
            lr *= 0.999
            
        if best_loss < 1e-6:
            # Check if we can round to integers
            U_int = np.round(best_U).astype(int)
            V_int = np.round(best_V).astype(int)
            W_int = np.round(best_W).astype(int)
            
            # Verify integer solution
            F_int = np.einsum('ijr,jkr,kir->ijk', U_int, V_int, W_int, optimize=True)
            if np.allclose(F_int, 1.0, atol=1e-9):
                U_sol = U_int.flatten().tolist()
                V_sol = V_int.flatten().tolist()
                W_sol = W_int.flatten().tolist()
                success = True
                break
            
            # Try half-integers?
            # If values are close to 0.5, 1.5 etc
            # We can just check if 2*best_U is close to int
            U_half = np.round(best_U * 2.0).astype(int) / 2.0
            V_half = np.round(best_V * 2.0).astype(int) / 2.0
            W_half = np.round(best_W * 2.0).astype(int) / 2.0
            
            F_half = np.einsum('ijr,jkr,kir->ijk', U_half, V_half, W_half, optimize=True)
            if np.allclose(F_half, 1.0, atol=1e-9):
                # Store as list of floats? Or scale?
                # The prompt says "integer/half-integer entries".
                # If I return floats, it should be fine if they are exact.
                U_sol = U_half.flatten().tolist()
                V_sol = V_half.flatten().tolist()
                W_sol = W_half.flatten().tolist()
                success = True
                break
                
    if success:
        return {"U": U_sol, "V": V_sol, "W": W_sol}
    else:
        # Fallback to Rank 28
        return _construct_rank_28()

def _construct_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    # Strassen 2x2 decomposition
    def strassen_terms():
        terms = []
        u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}; terms.append((u, v, w))
        u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}; terms.append((u, v, w))
        u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}; terms.append((u, v, w))
        u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}; terms.append((u, v, w))
        u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}; terms.append((u, v, w))
        u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}; terms.append((u, v, w))
        u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}; terms.append((u, v, w))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
