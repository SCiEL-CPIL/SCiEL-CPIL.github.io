import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

    def objective(x, T, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def greedy_merge_structured(U, V, W, max_steps=50):
        """Greedily merge columns that share index structure"""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            if U_curr.shape[1] <= 1:
                break
            
            best_merge = None
            best_error = float('inf')
            best_alpha = 0
            best_pair = None
            
            r = U_curr.shape[1]
            for a in range(r):
                for b in range(a + 1, r):
                    # Try several alpha values
                    for alpha in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                        U_new = U_curr.copy()
                        V_new = V_curr.copy()
                        W_new = W_curr.copy()
                        
                        U_new[:, a] = U_new[:, a] + alpha * U_new[:, b]
                        V_new[:, a] = V_new[:, a] + alpha * V_new[:, b]
                        W_new[:, a] = W_new[:, a] + alpha * W_new[:, b]
                        
                        # Remove column b
                        U_new = np.delete(U_new, b, axis=1)
                        V_new = np.delete(V_new, b, axis=1)
                        W_new = np.delete(W_new, b, axis=1)
                        
                        prod = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
                        error = np.sum((prod - T) ** 2)
                        if error < best_error:
                            best_error = error
                            best_alpha = alpha
                            best_pair = (a, b)
            
            if best_pair is not None and best_error < 1e-6:
                a, b = best_pair
                U_curr, V_curr, W_curr = U_curr.copy(), V_curr.copy(), W_curr.copy()
                U_curr[:, a] = U_curr[:, a] + best_alpha * U_curr[:, b]
                V_curr[:, a] = V_curr[:, a] + best_alpha * V_curr[:, b]
                W_curr[:, a] = W_curr[:, a] + best_alpha * W_curr[:, b]
                U_curr = np.delete(U_curr, b, axis=1)
                V_curr = np.delete(V_curr, b, axis=1)
                W_curr = np.delete(W_curr, b, axis=1)
            else:
                break
        
        return U_curr, V_curr, W_curr

    def structured_init(rank, seed):
        """Create structured initialization based on matrix multiplication structure"""
        rng = np.random.RandomState(seed)
        
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Group triples by shared indices to create structured columns
        # For each rank component, assign a pattern of (i,j,k) triples
        assigned = np.zeros((n, m, p), dtype=bool)
        
        for r in range(rank):
            # Choose a random subset of triples to assign to this rank
            # Prefer triples that share indices with already-assigned triples in this rank
            for _ in range(3):  # Try up to 3 times to find a good assignment
                # Pick a random starting triple
                unassigned = np.argwhere(~assigned)
                if len(unassigned) == 0:
                    break
                start_idx = rng.randint(len(unassigned))
                i0, j0, k0 = unassigned[start_idx]
                
                # Find all unassigned triples that share at least one index with (i0, j0, k0)
                candidates = []
                for idx in unassigned:
                    i, j, k = idx
                    if i == i0 or j == j0 or k == k0:
                        candidates.append(idx)
                
                if not candidates:
                    candidates = [unassigned[start_idx]]
                
                # Assign a random subset of candidates
                num_assign = min(len(candidates), rng.randint(1, 4))
                chosen = rng.choice(len(candidates), size=num_assign, replace=False)
                
                for c in chosen:
                    i, j, k = candidates[c]
                    if not assigned[i, j, k]:
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, r] = 1
                        V[v_idx, r] = 1
                        W[w_idx, r] = 1
                        assigned[i, j, k] = True
        
        # Add small random perturbation to break symmetry
        U += rng.randn(U_dim, rank) * 0.01
        V += rng.randn(V_dim, rank) * 0.01
        W += rng.randn(W_dim, rank) * 0.01
        
        return U, V, W

    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # First, try greedy merging from the trivial solution
    U_merged, V_merged, W_merged = greedy_merge_structured(U50, V50, W50, max_steps=30)
    if validate(U_merged, V_merged, W_merged):
        valid, U_s, V_s, W_s = snap_and_validate(U_merged, V_merged, W_merged)
        if valid:
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s, V_s, W_s

    # Try structured initializations and optimization for lower ranks
    for target_rank in range(48, 20, -1):
        if time.time() > deadline - 25:
            break
        if target_rank >= best_rank:
            continue
            
        found = False
        for seed in range(15):
            if time.time() > deadline - 15:
                break
                
            # Try structured initialization
            U_init, V_init, W_init = structured_init(target_rank, seed)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Optimization with tighter tolerances
                res = minimize(
                    objective, 
                    x0, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-10}
                )
                
                U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                V_opt = res.x[U_dim * target_rank : U_dim * rank + V_dim * target_rank].reshape(V_dim, target_rank)
                W_opt = res.x[U_dim * target_rank + V_dim * target_rank :].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
            except Exception:
                pass

        if found:
            continue

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
