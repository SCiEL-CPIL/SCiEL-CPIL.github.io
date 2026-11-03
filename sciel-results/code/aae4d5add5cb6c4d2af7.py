"""Matrix multiplication 3,4,5 rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    num_entries = n * m * p  # 60
    
    # Precompute index arrays
    i_idx = np.repeat(np.repeat(np.arange(n), m), p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def compute_error(U, V, W):
        products = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        sums = products.sum(axis=1)
        errors = sums - 1.0
        return np.sum(errors ** 2)
    
    def compute_gradient(U, V, W):
        products = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        sums = products.sum(axis=1)
        errors = sums - 1.0
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        contrib_U = 2.0 * errors[:, np.newaxis] * V[v_idx, :] * W[w_idx, :]
        contrib_V = 2.0 * errors[:, np.newaxis] * U[u_idx, :] * W[w_idx, :]
        contrib_W = 2.0 * errors[:, np.newaxis] * U[u_idx, :] * V[v_idx, :]
        
        for r in range(U.shape[1]):
            np.add.at(grad_U, u_idx, contrib_U[:, r])
            np.add.at(grad_V, v_idx, contrib_V[:, r])
            np.add.at(grad_W, w_idx, contrib_W[:, r])
        
        return grad_U, grad_V, grad_W
    
    def objective_and_grad(x, rank):
        U = x[:dim_u * rank].reshape(dim_u, rank)
        V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        
        err = compute_error(U, V, W)
        gU, gV, gW = compute_gradient(U, V, W)
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        return err, grad
    
    # Try to find lower rank decompositions
    # Use a range of ranks, starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    for rank in range(20, min(rank_sb + 1, 50)):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try multiple random initializations with different strategies
        for attempt in range(12):
            if time.time() > deadline:
                break
            
            seed = rank * 1000 + attempt * 7
            np.random.seed(seed)
            
            if attempt == 0:
                # Pure random
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
            elif attempt == 1:
                # Sparse random
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    u_sup = np.random.choice(dim_u, size=max(2, dim_u // 4), replace=False)
                    v_sup = np.random.choice(dim_v, size=max(2, dim_v // 4), replace=False)
                    w_sup = np.random.choice(dim_w, size=max(2, dim_w // 4), replace=False)
                    U_init[u_sup, r] = np.random.choice([-1, 1], size=len(u_sup))
                    V_init[v_sup, r] = np.random.choice([-1, 1], size=len(v_sup))
                    W_init[w_sup, r] = np.random.choice([-1, 1], size=len(w_sup))
            elif attempt == 2:
                # Structured: based on overlapping 2x2 blocks
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                # Use Strassen-like patterns for 2x2 blocks
                # For each pair of j indices, use rank-5 decomposition
                j_pairs = [(0,1), (1,2), (2,3)]
                col = 0
                for (j1, j2) in j_pairs:
                    # 5 rank-1 terms for 2x2 multiplication
                    for s in range(5):
                        if col >= rank:
                            break
                        # Strassen coefficients
                        if s == 0:
                            U_init[j1 * m + 0, col] = 1; V_init[j1 * p + 0, col] = 1
                            U_init[j1 * m + 1, col] = 1; V_init[j1 * p + 1, col] = 1
                        elif s == 1:
                            U_init[j1 * m + 1, col] = 1; V_init[j1 * p + 0, col] = 1
                            U_init[j2 * m + 0, col] = -1; V_init[j2 * p + 1, col] = 1
                        elif s == 2:
                            U_init[j1 * m + 0, col] = 1; V_init[j1 * p + 1, col] = 1
                            U_init[j2 * m + 1, col] = 1; V_init[j2 * p + 0, col] = 1
                        elif s == 3:
                            U_init[j2 * m + 1, col] = 1; V_init[j2 * p + 1, col] = 1
                            U_init[j1 * m + 0, col] = -1; V_init[j1 * p + 0, col] = 1
                        elif s == 4:
                            U_init[j2 * m + 0, col] = 1; V_init[j2 * p + 0, col] = 1
                            U_init[j1 * m + 1, col] = -1; V_init[j1 * p + 1, col] = 1
                        # W terms: need to map to correct output positions
                        # This is a rough initialization; optimization will fix it
                        for i in range(n):
                            for k in range(p):
                                if s == 0:
                                    W_init[k * n + i, col] = 1
                                elif s == 1:
                                    W_init[k * n + i, col] = 1
                                elif s == 2:
                                    W_init[k * n + i, col] = 1
                                elif s == 3:
                                    W_init[k * n + i, col] = 1
                                elif s == 4:
                                    W_init[k * n + i, col] = 1
                        col += 1
                # Add random perturbation to break symmetry
                U_init += 0.1 * np.random.randn(dim_u, rank)
                V_init += 0.1 * np.random.randn(dim_v, rank)
                W_init += 0.1 * np.random.randn(dim_w, rank)
            elif attempt == 3:
                # Small random with structure
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U_init[i * m + j, r] = 1
                    V_init[j * p + k, r] = 1
                    W_init[k * n + i, r] = 1
                    # Perturb neighbors
                    if r < rank - 1:
                        U_init[i * m + (j + 1) % m, r] = 0.5
                        V_init[(j + 1) % m * p + k, r] = 0.5
                        W_init[k * n + (i + 1) % n, r] = 0.5
                U_init += 0.2 * np.random.randn(dim_u, rank)
                V_init += 0.2 * np.random.randn(dim_v, rank)
                W_init += 0.2 * np.random.randn(dim_w, rank)
            elif attempt == 4:
                # Diagonal-like initialization
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    u_pos = r % dim_u
                    v_pos = r % dim_v
                    w_pos = r % dim_w
                    U_init[u_pos, r] = 1
                    V_init[v_pos, r] = 1
                    W_init[w_pos, r] = 1
                U_init += 0.5 * np.random.randn(dim_u, rank)
                V_init += 0.5 * np.random.randn(dim_v, rank)
                W_init += 0.5 * np.random.randn(dim_w, rank)
            elif attempt == 5:
                # Random with normalization
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
                for r in range(rank):
                    U_init[:, r] /= np.linalg.norm(U_init[:, r]) + 1e-10
                    V_init[:, r] /= np.linalg.norm(V_init[:, r]) + 1e-10
                    W_init[:, r] /= np.linalg.norm(W_init[:, r]) + 1e-10
                U_init *= 2
                V_init *= 2
                W_init *= 2
            elif attempt == 6:
                # Very sparse
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    u_pos = np.random.randint(0, dim_u)
                    v_pos = np.random.randint(0, dim_v)
                    w_pos = np.random.randint(0, dim_w)
                    U_init[u_pos, r] = np.random.choice([-1, 1])
                    V_init[v_pos, r] = np.random.choice([-1, 1])
                    W_init[w_pos, r] = np.random.choice([-1, 1])
            elif attempt == 7:
                # Block diagonal structure
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                block_size = max(1, rank // 4)
                for b in range(4):
                    start = b * block_size
                    end = min(start + block_size, rank)
                    for r in range(start, end):
                        i = b % n
                        j = (r - start) % m
                        k = (r - start) % p
                        U_init[i * m + j, r] = 1
                        V_init[j * p + k, r] = 1
                        W_init[k * n + i, r] = 1
                U_init += 0.3 * np.random.randn(dim_u, rank)
                V_init += 0.3 * np.random.randn(dim_v, rank)
                W_init += 0.3 * np.random.randn(dim_w, rank)
            elif attempt == 8:
                # Random with half-integer bias
                U_init = np.random.choice([0, 0.5, 1, 1.5, -0.5, -1, -1.5], size=(dim_u, rank))
                V_init = np.random.choice([0, 0.5, 1, 1.5, -0.5, -1, -1.5], size=(dim_v, rank))
                W_init = np.random.choice([0, 0.5, 1, 1.5, -0.5, -1, -1.5], size=(dim_w, rank))
            elif attempt == 9:
                # Two-scale random
                U_init = np.random.randn(dim_u, rank) * 0.5 + np.random.choice([-1, 0, 1], size=(dim_u, rank))
                V_init = np.random.randn(dim_v, rank) * 0.5 + np.random.choice([-1, 0, 1], size=(dim_v, rank))
                W_init = np.random.randn(dim_w, rank) * 0.5 + np.random.choice([-1, 0, 1], size=(dim_w, rank))
            elif attempt == 10:
                # Random with column scaling
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
                for r in range(rank):
                    scale = np.random.uniform(0.5, 2.0)
                    U_init[:, r] *= scale
                    V_init[:, r] *= scale
                    W_init[:, r] /= scale * scale
            else:
                # Pure random with different scale
                U_init = np.random.randn(dim_u, rank) * np.random.uniform(0.5, 3.0)
                V_init = np.random.randn(dim_v, rank) * np.random.uniform(0.5, 3.0)
                W_init = np.random.randn(dim_w, rank) * np.random.uniform(0.5, 3.0)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Check initial error
            init_err = compute_error(U_init, V_init, W_init)
            if init_err > 1e6:
                continue
            
            # Use L-BFGS-B for gradient-based optimization
            try:
                result = minimize(objective_and_grad, x0, args=(rank,), method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12,
                                           'maxls': 100, 'maxcor': 20})
                
                if result.fun < 1e-10:
                    U_opt = result.x[:dim_u * rank].reshape(dim_u, rank)
                    V_opt = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    W_opt = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    max_err = compute_error(U_rounded, V_rounded, W_rounded)
                    if max_err < 1e-9:
                        best_result = (U_rounded, V_rounded, W_rounded)
                        best_rank = rank
                        found = True
                        break
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    max_err_int = compute_error(U_int, V_int, W_int)
                    if max_err_int < 1e-9:
                        best_result = (U_int, V_int, W_int)
                        best_rank = rank
                        found = True
                        break
                    
                    # Try rounding to quarters then to halves
                    U_quarter = np.round(U_opt * 4) / 4
                    V_quarter = np.round(V_opt * 4) / 4
                    W_quarter = np.round(W_opt * 4) / 4
                    max_err_q = compute_error(U_quarter, V_quarter, W_quarter)
                    if max_err_q < 1e-9:
                        # Check if quarters can be expressed as halves (they always can if even)
                        # But we need half-integers, so quarters with odd numerator are not valid
                        # Try to adjust
                        U_adj = np.round(U_quarter * 2) / 2
                        V_adj = np.round(V_quarter * 2) / 2
                        W_adj = np.round(W_quarter * 2) / 2
                        max_err_adj = compute_error(U_adj, V_adj, W_adj)
                        if max_err_adj < 1e-9:
                            best_result = (U_adj, V_adj, W_adj)
                            best_rank = rank
                            found = True
                            break
                            
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

