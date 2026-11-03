import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (5s buffer)

    # Convert to int64 arrays for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    best_p = None
    best_cost = np.iinfo(np.int64).max

    def compute_cost(p):
        """Compute the QAP cost for permutation p using int64 arithmetic."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return int(np.sum(a * b[np.ix_(p, p)]))

    def delta_cost_swap(p, i, j):
        """
        Calculate the change in cost when swapping locations of facilities i and j.
        p[i] = u, p[j] = v. After swap, p'[i] = v, p'[j] = u.
        
        Uses vectorized dot products for O(n) computation.
        """
        u = p[i]
        v = p[j]
        
        if u == v:
            return 0
        
        # Before contributions:
        # Row i: sum_n a[i,n] b[u, p[n]]
        # Row j: sum_n a[j,n] b[v, p[n]]
        # Col i: sum_m a[m,i] b[p[m], u]
        # Col j: sum_m a[m,j] b[p[m], v]
        # Double counts: a[i,i]b[u,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,i]b[v,u]
        
        # After contributions:
        # Row i: sum_n a[i,n] b[v, p[n]]
        # Row j: sum_n a[j,n] b[u, p[n]]
        # Col i: sum_m a[m,i] b[p[m], v]
        # Col j: sum_m a[m,j] b[p[m], u]
        # Double counts: a[i,i]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
        
        before_i = np.dot(a[i, :], b[u, p])
        before_j = np.dot(a[j, :], b[v, p])
        before_ci = np.dot(a[:, i], b[p, u])
        before_cj = np.dot(a[:, j], b[p, v])
        before_dd = a[i,i]*b[u,u] + a[j,j]*b[v,v] + a[i,j]*b[u,v] + a[j,i]*b[v,u]
        
        after_i = np.dot(a[i, :], b[v, p])
        after_j = np.dot(a[j, :], b[u, p])
        after_ci = np.dot(a[:, i], b[p, v])
        after_cj = np.dot(a[:, j], b[p, u])
        after_dd = a[i,i]*b[v,v] + a[j,j]*b[u,u] + a[i,j]*b[v,u] + a[j,i]*b[u,v]
        
        delta = (after_i + after_j + after_ci + after_cj - after_dd) - \
                (before_i + before_j + before_ci + before_cj - before_dd)
        
        return int(delta)

    def local_search_2opt(p, max_time=None):
        """Perform 2-opt local search with time limit."""
        p = p.copy()
        current_cost = compute_cost(p)
        improved = True
        passes = 0
        
        while improved:
            improved = False
            passes += 1
            for i in range(n):
                if max_time and time.time() > max_time:
                    return p, current_cost
                for j in range(i+1, n):
                    delta = delta_cost_swap(p, i, j)
                    if delta < 0:
                        # Swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    def linex_init(a, b):
        """
        Linex initialization: Use linear assignment to create initial permutation.
        This is a heuristic that matches facilities to locations based on 
        aggregated flow-distance interactions.
        """
        # Compute for each facility i and location k:
        # score[i,k] = sum_j a[i,j] * b[k, j'] where j' is some proxy
        # A simpler approach: use the sum of flows and distances
        # facility_i_weight = sum_j a[i,j]
        # location_k_weight = sum_j b[k,j]
        # Then solve assignment to minimize sum_i facility_i_weight[i] * location_k_weight[p[i]]
        
        fac_weights = a.sum(axis=1).astype(np.float64)
        loc_weights = b.sum(axis=1).astype(np.float64)
        
        # Solve assignment problem to minimize sum_i fac_weights[i] * loc_weights[p[i]]
        # This is equivalent to minimizing sum_i fac_weights[i] * loc_weights[p[i]]
        # Create cost matrix C[i,k] = fac_weights[i] * loc_weights[k]
        C = np.outer(fac_weights, loc_weights)
        
        # Use scipy's linear_sum_assignment
        _, p = linear_sum_assignment(C)
        return p.astype(np.int64)

    def random_restart_init(n, rng):
        """Generate a random permutation."""
        return rng.permutation(n).astype(np.int64)

    # Generate initial solutions
    # 1. Linex initialization
    p_linex = linex_init(a, b)
    p_ls, cost_ls = local_search_2opt(p_linex, max_time=deadline - 5)
    best_p = p_ls.copy()
    best_cost = cost_ls

    # 2. Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_ls, cost_ls = local_search_2opt(p_rank, max_time=deadline - 5)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # 3. Try multiple random restarts
    random.seed(42)
    np_random = np.random.RandomState(42)
    
    num_restarts = 0
    while time.time() < deadline - 10:
        p_rand = random_restart_init(n, np_random)
        p_ls, cost_ls = local_search_2opt(p_rand, max_time=deadline - 5)
        num_restarts += 1
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # Final check
    if best_p is None:
        # Fallback to rank matching if no search completed
        best_p = p_rank
    
    return best_p.tolist()
