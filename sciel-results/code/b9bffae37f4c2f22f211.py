"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(i, j, p):
        """Compute delta for swapping facilities i and j.
        Assumes symmetric a and b."""
        u = p[i]
        v = p[j]
        
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        diff_a = ai - aj
        
        # Sum over all k
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        
        # Subtract k=i term: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j term: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        # Self-term correction: 2*(a[i,i]-a[j,j])*(b[v,v]-b[u,u])
        self_term = 2 * (ai[i] - aj[j]) * (bv[v] - bu[u])
        
        delta = 2 * (sum_all - term_i - term_j) + self_term
        return int(delta)
    
    def local_search(p, current_cost, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost
    
    def get_3cycle_delta(i, j, k, p):
        """Cost delta for 3-cycle: i->p[j], j->p[k], k->p[i]"""
        u, v, w = p[i], p[j], p[k]
        # New locations
        new_loc = {i: v, j: w, k: u}
        
        # Internal: 9 pairs
        internal = 0
        for x in (i, j, k):
            for y in (i, j, k):
                internal += a[x, y] * (b[new_loc[x], new_loc[y]] - b[p[x], p[y]])
        
        # Cross: x in {i,j,k}, y not in {i,j,k}
        cross = 0
        for x in (i, j, k):
            nl = new_loc[x]
            ol = p[x]
            # a[x, y] * (b[nl, p[y]] - b[ol, p[y]]) for y not in S
            mask = np.ones(n, dtype=bool)
            mask[[i, j, k]] = False
            cross += np.dot(a[x, mask], b[nl, p[mask]] - b[ol, p[mask]])
            # a[y, x] * (b[p[y], nl] - b[p[y], ol])
            cross += np.dot(a[mask, x], b[p[mask], nl] - b[p[mask], ol])
        
        return int(internal + cross)
    
    def local_search_3opt(p, current_cost, max_iters=20, samples_per_iter=2000):
        """3-opt local search with random sampling"""
        rng_local = np.random.RandomState(123)
        for _ in range(max_iters):
            found = False
            best_delta = 0
            best_triplet = None
            for _ in range(samples_per_iter):
                i, j, k = rng_local.choice(n, size=3, replace=False)
                delta = get_3cycle_delta(i, j, k, p)
                if delta < best_delta:
                    best_delta = delta
                    best_triplet = (i, j, k)
            if best_triplet is not None:
                i, j, k = best_triplet
                u, v, w = p[i], p[j], p[k]
                p[i], p[j], p[k] = v, w, u
                current_cost += best_delta
                found = True
            if not found:
                break
        return p, current_cost
    
    # Generate multiple initial solutions and pick the best
    def hungarian_init():
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        cost_matrix = np.outer(flow_sums, dist_sums)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    def random_init():
        return np.random.permutation(n).astype(np.int64)
    
    def spectral_init():
        # Use eigenvectors of row sums
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        # Normalize and use as cost
        fs_norm = flow_sums / (np.max(flow_sums) + 1e-10)
        ds_norm = dist_sums / (np.max(dist_sums) + 1e-10)
        cost_matrix = np.outer(fs_norm, ds_norm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    # Try multiple initializations
    candidates = [hungarian_init(), spectral_init(), random_init()]
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = calc_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    start_time = time.time()
    timeout = 115.0
    
    # ILS with simulated annealing perturbation
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < timeout:
        # 2-swap local search
        p, current_cost = local_search(p, current_cost, max_iters=30)
        
        # 3-opt local search to escape 2-opt local optima
        if time.time() - start_time < timeout - 10:
            p, current_cost = local_search_3opt(p, current_cost, max_iters=10, samples_per_iter=1500)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() - start_time > timeout - 5:
            break
        
        # Perturbation: random swaps
        num_swaps = 3 + rng.randint(7)
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            delta = get_delta(i, j, p)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
