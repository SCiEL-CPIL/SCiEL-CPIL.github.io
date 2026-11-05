import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def strassen_2x2(oi, oj, ok):
        add_term({(oi, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1, (oi+1, ok+1): 1})
        add_term({(oi+1, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi+1, ok): 1, (oi+1, ok+1): -1})
        add_term({(oi, oj): 1}, {(oj, ok+1): 1, (oj+1, ok+1): -1}, {(oi, ok+1): 1, (oi+1, ok+1): 1})
        add_term({(oi+1, oj+1): 1}, {(oj+1, ok): 1, (oj, ok): -1}, {(oi, ok): 1, (oi+1, ok): 1})
        add_term({(oi, oj): 1, (oi, oj+1): 1}, {(oj+1, ok+1): 1}, {(oi, ok): -1, (oi, ok+1): 1})
        add_term({(oi+1, oj): 1, (oi, oj): -1}, {(oj, ok): 1, (oj, ok+1): 1}, {(oi+1, ok+1): 1})
        add_term({(oi, oj+1): 1, (oi+1, oj+1): -1}, {(oj+1, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1})

    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        strassen_2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def strassen_4x4(oi, oj, ok):
        _strassen_super(oi, oj, ok, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        _strassen_super(oi, oj, ok, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        _strassen_super(oi, oj, ok, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        _strassen_super(oi, oj, ok, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        _strassen_super(oi, oj, ok, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        _strassen_super(oi, oj, ok, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        _strassen_super(oi, oj, ok, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})

    # Build initial rank-77 decomposition
    if _time_remaining() > 20.0:
        strassen_4x4(0, 0, 0)
        strassen_2x2(0, 0, 4)
        strassen_2x2(0, 2, 4)
        strassen_2x2(2, 0, 4)
        strassen_2x2(2, 2, 4)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    U_arr = np.array(U_final, dtype=np.float64)
    V_arr = np.array(V_final, dtype=np.float64)
    W_arr = np.array(W_final, dtype=np.float64)
    
    # Verify the initial decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][column] = 1
                                V_fb[j2 * p + k2][column] = 1
                                W_fb[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank via numerical optimization
    # Use BFGS-like approach: minimize sum of squared residuals for each (i,j,k)
    # while trying to make some columns near-zero
    
    def compute_residuals(U, V, W):
        """Compute residuals for all (i,j,k) triples."""
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    residuals[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :]) - 1.0
        return residuals
    
    def objective_and_grad(params, U_shape, V_shape, W_shape):
        """Unpack params, compute objective and gradient."""
        r = params.shape[0] // (U_shape[0] + V_shape[0] + W_shape[0])
        U = params[:r*U_shape[0]].reshape(U_shape[0], r)
        V = params[r*U_shape[0]:r*U_shape[0]+r*V_shape[0]].reshape(V_shape[0], r)
        W = params[r*U_shape[0]+r*V_shape[0]:].reshape(W_shape[0], r)
        
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    vw = V[v_idx, :] * W[w_idx, :]
                    res = np.dot(U[u_idx, :], vw) - 1.0
                    obj += res * res
                    grad_U[u_idx, :] += 2.0 * res * vw
                    grad_V[v_idx, :] += 2.0 * res * U[u_idx, :] * W[w_idx, :]
                    grad_W[w_idx, :] += 2.0 * res * U[u_idx, :] * V[v_idx, :]
        
        # Add penalty for non-zero entries in columns we want to eliminate
        # (this encourages rank reduction)
        penalty = 0.0
        pen_U = np.zeros_like(U)
        pen_V = np.zeros_like(V)
        pen_W = np.zeros_like(W)
        
        # Try to eliminate the last few columns
        n_elim = max(0, r - 70)
        if n_elim > 0:
            for c in range(r - n_elim, r):
                penalty += np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2)
                pen_U[:, c] += 2.0 * U[:, c]
                pen_V[:, c] += 2.0 * V[:, c]
                pen_W[:, c] += 2.0 * W[:, c]
        
        total_obj = obj + 0.1 * penalty
        grad = np.concatenate([
            (grad_U + 0.1 * pen_U).flatten(),
            (grad_V + 0.1 * pen_V).flatten(),
            (grad_W + 0.1 * pen_W).flatten()
        ])
        
        return total_obj, grad
    
    # Try optimization with different target ranks
    best_result = None
    best_score = rank
    
    for target_rank in [76, 75, 74, 73, 72, 71, 70]:
        if _time_remaining() < 15.0:
            break
        
        # Initialize with the current decomposition, zero out extra columns
        U_opt = U_arr.copy()
        V_opt = V_arr.copy()
        W_opt = W_arr.copy()
        
        # Pad or truncate to target_rank
        if target_rank < rank:
            U_opt = U_opt[:, :target_rank]
            V_opt = V_opt[:, :target_rank]
            W_opt = W_opt[:, :target_rank]
        elif target_rank > rank:
            U_opt = np.zeros((n*m, target_rank))
            V_opt = np.zeros((m*p, target_rank))
            W_opt = np.zeros((p*n, target_rank))
            U_opt[:, :rank] = U_arr
            V_opt[:, :rank] = V_arr
            W_opt[:, :rank] = W_arr
        
        # Add small random perturbation to avoid exact zero
        np.random.seed(42 + target_rank)
        U_opt += np.random.randn(*U_opt.shape) * 0.01
        V_opt += np.random.randn(*V_opt.shape) * 0.01
        W_opt += np.random.randn(*W_opt.shape) * 0.01
        
        params = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
        
        # Simple gradient descent with momentum
        lr = 0.001
        momentum = 0.9
        v_mom = np.zeros_like(params)
        
        for step in range(2000):
            if _time_remaining() < 15.0:
                break
            
            obj, grad = objective_and_grad(params, (n*m, target_rank), (m*p, target_rank), (p*n, target_rank))
            
            v_mom = momentum * v_mom - lr * grad
            params = params + v_mom
            
            if step % 200 == 0:
                U_cur = params[:target_rank*n*m].reshape(n*m, target_rank)
                V_cur = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(m*p, target_rank)
                W_cur = params[target_rank*(n*m+m*p):].reshape(p*n, target_rank)
                
                # Check if we can drop some columns
                col_norms = np.sqrt(np.sum(U_cur**2, axis=0) + np.sum(V_cur**2, axis=0) + np.sum(W_cur**2, axis=0))
                small_cols = np.where(col_norms < 0.01)[0]
                
                if len(small_cols) > 0:
                    # Try dropping the smallest column
                    drop_idx = small_cols[np.argmin(col_norms[small_cols])]
                    keep = [c for c in range(target_rank) if c != drop_idx]
                    
                    U_test = U_cur[:, keep]
                    V_test = V_cur[:, keep]
                    W_test = W_cur[:, keep]
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                total = np.dot(U_test[u_idx, :], V_test[v_idx, :] * W_test[w_idx, :])
                                if abs(total - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        new_rank = len(keep)
                        if new_rank < best_score:
                            best_score = new_rank
                            U_final = U_test.tolist()
                            V_final = V_test.tolist()
                            W_final = W_test.tolist()
                        break
        
        if _time_remaining() < 15.0:
            break
    
    # Final verification
    if best_result is None:
        U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
        V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
        W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    U_arr = np.array(U_final, dtype=np.float64)
    V_arr = np.array(V_final, dtype=np.float64)
    W_arr = np.array(W_final, dtype=np.float64)
    
    final_rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][column] = 1
                                V_fb[j2 * p + k2][column] = 1
                                W_fb[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_final, "V": V_final, "W": W_final}
