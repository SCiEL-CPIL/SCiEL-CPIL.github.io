"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Precompute some useful quantities
    # a is flow matrix, b is distance matrix
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    # Spectral initialization
    try:
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        # Use multiple eigenvectors for better initialization
        # Take top 3 eigenvectors
        v_a = eigvecs_a[:, -3:]
        v_b = eigvecs_b[:, -3:]
        
        # Project onto principal components and combine
        facility_scores = v_a @ np.array([1.0, 0.5, 0.25])
        location_scores = v_b @ np.array([1.0, 0.5, 0.25])
        
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
    except Exception:
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
    
    best_p = p.copy()
    best_cost = calc_cost(best_p)
    
    # Efficient delta computation for swap (i, j)
    # When we swap p[i] and p[j], the cost change involves:
    # 1. The interaction between i and j themselves
    # 2. The interaction of i with all other k (where k != i,j)
    # 3. The interaction of j with all other k (where k != i,j)
    
    def get_swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # The change in cost when swapping locations u and v for facilities i and j
        # Original: a[i,j]*b[u,v] + a[j,i]*b[v,u] + sum_{k!=i,j}(a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]])
        # New: a[i,j]*b[v,u] + a[j,i]*b[u,v] + sum_{k!=i,j}(a[i,k]*b[v,p[k]] + a[j,k]*b[u,p[k]])
        
        # For symmetric matrices, a[i,j]=a[j,i] and b[u,v]=b[v,u]
        # So the i-j interaction doesn't change.
        
        # Change = sum_{k!=i,j} [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])]
        #         + sum_{k!=i,j} [(a[j,k]-a[i,k])*(b[u,p[k]]-b[v,p[k]])]
        #         = 2 * sum_{k!=i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
        
        # Vectorized computation
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        # b[v, p[k]] for all k
        bv_p = bv[p]
        bu_p = bu[p]
        
        diff_a = ai - aj
        diff_b = bv_p - bu_p
        
        # Full sum
        total = np.dot(diff_a, diff_b)
        
        # Subtract k=i term: (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j term: (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        delta = 2 * (total - term_i - term_j)
        return int(delta)
    
    # VND (Variable Neighborhood Descent) with multiple neighborhood structures
    def vnd_search(p, current_cost, time_limit):
        """
        Variable Neighborhood Descent:
        - N1: 2-swap neighborhood (standard)
        - N2: 3-swap neighborhood (try cycling 3 facilities)
        - N3: Block swap (swap a block of consecutive facilities)
        """
        improved = True
        iter_count = 0
        max_iters = 500
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Check time
            if time.time() > time_limit:
                break
            
            # N1: 2-swap moves
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    delta = get_swap_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        
        return p, current_cost
    
    # 3-cycle move: cyclically permute p[i], p[j], p[k]
    def get_3cycle_delta(i, j, k, p):
        """Compute delta for cycling p[i]->p[j]->p[k]->p[i]"""
        u = p[i]
        v = p[j]
        w = p[k]
        
        # Original cost contributions involving i, j, k
        # This is complex, so we compute directly
        # Old: sum of a[i,*]*b[u,*] + a[j,*]*b[v,*] + a[k,*]*b[w,*] for * in {i,j,k}
        # plus interactions with others
        
        # Simpler: compute cost difference directly
        # Save old positions
        old_p = p.copy()
        
        # Apply 3-cycle: i gets v's loc, j gets w's loc, k gets u's loc
        p[i] = v
        p[j] = w
        p[k] = u
        
        # Compute new cost
        new_cost = calc_cost(p)
        
        # Restore
        p[:] = old_p
        
        return new_cost - current_cost
    
    # Simulated Annealing with VND
    def anneal_with_vnd(p, current_cost, start_time, timeout):
        """
        Hybrid: use VND for local search, then perturb with random moves,
        accepting worse moves with probability based on temperature.
        """
        T = max(1.0, current_cost * 0.01)  # Initial temperature
        T_min = max(1.0, current_cost * 0.0001)
        cooling_rate = 0.995
        
        best_p = p.copy()
        best_cost = current_cost
        
        while time.time() - start_time < timeout and T > T_min:
            # Perform VND local search
            p, current_cost = vnd_search(p, current_cost, time.time() + 5.0)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            # Check time
            if time.time() - start_time > timeout:
                break
            
            # Perturbation: random 2-swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = get_swap_delta(i, j, p)
            
            # Metropolis acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
            
            # Cool
            T *= cooling_rate
        
        return best_p, best_cost
    
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Initial VND
    p, current_cost = vnd_search(p, calc_cost(p), start_time + 20.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Phase 2: Annealing with VND restarts
    while time.time() - start_time < timeout:
        # Perturb the current best
        p = best_p.copy()
        current_cost = best_cost
        
        # Apply several random swaps as perturbation
        num_perturb = np.random.randint(3, 10)
        for _ in range(num_perturb):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                delta = get_swap_delta(i, j, p)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        
        # Run annealing with VND from this perturbed state
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 10.0:
            break
        
        p, current_cost = anneal_with_vnd(p, current_cost, time.time(), remaining_time * 0.8)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Final polish: run VND on the best solution found
    remaining = timeout - (time.time() - start_time)
    if remaining > 5.0:
        best_p, best_cost = vnd_search(best_p, best_cost, time.time() + min(remaining - 1.0, 10.0))
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
