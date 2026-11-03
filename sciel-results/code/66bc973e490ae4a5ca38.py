# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s/150s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # --- Initialization strategies ---
    
    # 1. Spectral initialization
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]

    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]

    fac_order = np.argsort(-v_a, kind="stable")
    loc_order = np.argsort(v_b, kind="stable")
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[fac_order] = loc_order
    cost_spectral = compute_cost(p_spectral)

    # 2. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)

    # 3. Hungarian algorithm initialization (Linear Surrogate)
    C_hung = np.zeros((n, n))
    for i in range(n):
        C_hung[i, :] = np.dot(a[i, :], b)
    
    row_ind, col_ind = linear_sum_assignment(C_hung)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    cost_hung = compute_cost(p_hung)

    # Select best initial
    candidates = [
        (p_spectral, cost_spectral),
        (p_rank, cost_rank),
        (p_hung, cost_hung)
    ]
    p, best_cost = min(candidates, key=lambda x: x[1])

    # --- Precompute for fast delta ---
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    
    def get_delta(p, i, j):
        """Compute cost delta for swapping facilities i and j (locations u=p[i], v=p[j])."""
        u = p[i]
        v = p[j]
        
        # The cost is sum_{k,l} a[k,l] * b[p[k], p[l]]
        # After swap: p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
        # Delta = sum_{k,l} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
        # Only terms where k or l is i or j change.
        
        # Terms involving i in first index:
        # sum_l a[i,l] * (b[v, p'[l]] - b[u, p[l]])
        # l=i: a[i,i]*(b[v,v]-b[u,u])
        # l=j: a[i,j]*(b[v,u]-b[u,v])
        # l!=i,j: a[i,l]*(b[v,p[l]]-b[u,p[l]])
        
        # Terms involving i in second index:
        # sum_k a[k,i] * (b[p'[k], v] - b[p[k], u])
        # k=i: a[i,i]*(b[v,v]-b[u,u])
        # k=j: a[j,i]*(b[u,v]-b[v,u])
        # k!=i,j: a[k,i]*(b[p[k],v]-b[p[k],u])
        
        # Terms involving j in first index:
        # sum_l a[j,l] * (b[u, p'[l]] - b[v, p[l]])
        # l=i: a[j,i]*(b[u,v]-b[v,u])
        # l=j: a[j,j]*(b[u,u]-b[v,v])
        # l!=i,j: a[j,l]*(b[u,p[l]]-b[v,p[l]])
        
        # Terms involving j in second index:
        # sum_k a[k,j] * (b[p'[k], u] - b[p[k], v])
        # k=i: a[i,j]*(b[v,u]-b[u,v])
        # k=j: a[j,j]*(b[u,u]-b[v,v])
        # k!=i,j: a[k,j]*(b[p[k],u]-b[p[k],v])
        
        # Direct terms (k,l in {i,j}):
        direct = (2*a[i,i]*(b[v,v]-b[u,u]) + 
                  2*a[j,j]*(b[u,u]-b[v,v]) +
                  2*a[i,j]*(b[v,u]-b[u,v]) +
                  2*a[j,i]*(b[u,v]-b[v,u]))
        
        # Cross terms (k,l not both in {i,j}, but one of k or l is i or j)
        # For k!=i,j, l!=i,j: no change
        # For k=i, l!=i,j: a[i,l]*(b[v,p[l]]-b[u,p[l]])
        # For k=j, l!=i,j: a[j,l]*(b[u,p[l]]-b[v,p[l]])
        # For l=i, k!=i,j: a[k,i]*(b[p[k],v]-b[p[k],u])
        # For l=j, k!=i,j: a[k,j]*(b[p[k],u]-b[p[k],v])
        
        # Vectorized computation:
        # sum_{l!=i,j} a[i,l]*(b[v,p[l]]-b[u,p[l]]) + a[j,l]*(b[u,p[l]]-b[v,p[l]])
        # = sum_{l!=i,j} (a[i,l]-a[j,l])*(b[v,p[l]]-b[u,p[l]])
        # = (a[i,:]-a[j,:]) @ (b[v,p]-b[u,p]) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,v]-b[u,v]) + a[j,i]*(b[u,u]-b[v,u]) + a[j,j]*(b[u,v]-b[v,v])
        
        bv_p = b[v, p]
        bu_p = b[u, p]
        bp_v = b[p, v]
        bp_u = b[p, u]
        
        # Row cross: sum_l (a[i,l]-a[j,l])*(b[v,p[l]]-b[u,p[l]])
        row_diff = a[i] - a[j]
        b_row_diff = bv_p - bu_p
        row_cross_full = np.dot(row_diff, b_row_diff)
        # Subtract l=i and l=j terms
        row_cross = row_cross_full - (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) - (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        
        # Col cross: sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        col_diff = a[:, i] - a[:, j]
        b_col_diff = bp_v - bp_u
        col_cross_full = np.dot(col_diff, b_col_diff)
        # Subtract k=i and k=j terms
        col_cross = col_cross_full - (a[i,i]-a[i,j])*(b[u,v]-b[u,u]) - (a[j,i]-a[j,j])*(b[v,v]-b[u,v])
        
        return direct + row_cross + col_cross

    def local_search_first_improvement(p, current_cost, max_passes=50):
        """Local search with first improvement for speed."""
        p = p.copy()
        for _ in range(max_passes):
            if time.time() - t_start > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                for j in range(i+1, n):
                    delta = get_delta(p, i, j)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return p, current_cost

    def local_search_best_improvement(p, current_cost):
        """Local search with best improvement per pass."""
        p = p.copy()
        while True:
            if time.time() - t_start > deadline:
                break
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                for j in range(i+1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
            else:
                break
        return p, current_cost

    # Run best-improvement local search on best initial
    p, best_cost = local_search_best_improvement(p, best_cost)

    # --- Iterated Local Search ---
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 15
    perturbation_size = 3

    while time.time() - t_start < deadline - 2.0:
        # Adaptive perturbation
        p_pert = p.copy()
        for _ in range(perturbation_size):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Use first-improvement for perturbed solution (faster)
        p_new, cost_new = local_search_first_improvement(p_pert, compute_cost(p_pert))
        
        if cost_new < best_cost:
            # Polish with best-improvement
            p_new, cost_new = local_search_best_improvement(p_new, cost_new)
            p = p_new
            best_cost = cost_new
            no_improve_count = 0
            # Reduce perturbation on success
            perturbation_size = max(2, perturbation_size - 1)
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Stronger perturbation
                no_improve_count = 0
                perturbation_size = min(8, perturbation_size + 3)
                p_pert = p.copy()
                for _ in range(perturbation_size):
                    i, j = rng.choice(n, 2, replace=False)
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                p_new, cost_new = local_search_first_improvement(p_pert, compute_cost(p_pert))
                p_new, cost_new = local_search_best_improvement(p_new, cost_new)
                if cost_new < best_cost:
                    p = p_new
                    best_cost = cost_new

    return p.tolist()
# EVOLVE-BLOCK-END
