"""Combined solver: Hungarian initialization + vectorized best-improvement local search + ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s kill

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        Uses full vectorized computation over all k, subtracting i and j terms.
        """
        u = p[i]
        v = p[j]
        
        # Direct pair term: a[i,j]*b[u,v] + a[j,i]*b[v,u] -> a[i,j]*b[v,u] + a[j,i]*b[u,v]
        # Delta = a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        #       = (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms: a[i,i]*b[u,u] + a[j,j]*b[v,v] -> a[i,i]*b[v,v] + a[j,j]*b[u,u]
        # Delta = a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v])
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        
        # Precompute vectors
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        # b[v, p] is a vector of length n
        # b[u, p] is a vector of length n
        # diff_b_row_k = b[v, p[k]] - b[u, p[k]]
        diff_b_row = b[v, p] - b[u, p]
        
        # b[p, v] is a vector of length n
        # b[p, u] is a vector of length n
        # diff_b_col_k = b[p[k], v] - b[p[k], u]
        diff_b_col = b[p, v] - b[p, u]
        
        # Sum over all k
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        # Term 1, k=i: (a[i,i]-a[j,i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i,i]-a[j,i]) * (b[v, u] - b[u, u])
        sum_T1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # Term 2, k=i: (a[i,i]-a[i,j]) * (b[p[i], v] - b[p[i], u]) = (a[i,i]-a[i,j]) * (b[u, v] - b[u, u])
        sum_T2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        
        # Subtract k=j terms
        # Term 1, k=j: (a[i,j]-a[j,j]) * (b[v, p[j]] - b[u, p[j]]) = (a[i,j]-a[j,j]) * (b[v, v] - b[u, v])
        sum_T1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        # Term 2, k=j: (a[j,i]-a[j,j]) * (b[p[j], v] - b[p[j], u]) = (a[j,i]-a[j,j]) * (b[v, v] - b[v, u])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        delta += sum_T1 + sum_T2
        return int(delta)

    def local_search(p, current_cost):
        """
        Best-improvement local search.
        Iterates through all pairs (i, j) to find the single best improving swap.
        Repeats until no improving swap exists or timeout.
        """
        while True:
            if time.time() - start_time > timeout:
                break
            
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # To speed up, we can sample pairs or use first-improvement if time is tight.
            # But best-improvement is stronger. For n=100, 4950 pairs.
            # Each delta calc is O(n) vectorized. ~5000 * 100 ops = 500k ops per pass.
            # This is fast enough in NumPy.
            
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
            else:
                break
        
        return p, current_cost

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on surrogate cost.
        C[i, j] = -a_row_sums[i] * b_row_sums[j]
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        # We want to maximize sum a_row_sums[i] * b_row_sums[p[i]]
        # So minimize sum -a_row_sums[i] * b_row_sums[p[i]]
        C = -np.outer(a_row_sums, b_row_sums)
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # 1. Initialization
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Local Search
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    # Use a random state for reproducibility within the run, but different each call?
    # The task says "seed/value", implying the evaluator might set a seed or run multiple times.
    # We should use np.random with default seed or allow it to be non-deterministic if not set.
    # However, for stable scoring in this context, let's just use the global np.random state.
    
    while time.time() - start_time < timeout - 2.0:
        # Perturbation
        if no_improve_count < 5:
            num_swaps = 4
        elif no_improve_count < 15:
            num_swaps = 8
        else:
            num_swaps = 15
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Restart from best
            p = best_p.copy()
            current_cost = best_cost
            
        # Check timeout
        if time.time() - start_time > timeout - 5.0:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
