import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    def compute_cost(p):
        """Compute QAP cost for permutation p."""
        return np.sum(a * b[np.ix_(p, p)])

    def spectral_init():
        """Initialize using spectral method: match eigenvectors of A and B."""
        # Use the leading eigenvectors to create a matching
        # For QAP, we want facilities with high flow to be near locations with high distance
        # Use the sum of row sums as a simple heuristic, enhanced with spectral info
        
        # Compute row sums
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        
        # Sort facilities by flow (descending) and locations by distance (ascending)
        heavy = np.argsort(-a_sums, kind="stable")
        central = np.argsort(b_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def local_search(p, max_time=None):
        """
        Fast local search using precomputed row/col contributions.
        Swap p[i] and p[j], compute delta in O(n) instead of O(n^2).
        """
        current_p = p.copy()
        current_cost = compute_cost(current_p)
        
        # Precompute B_perm and row/col contributions
        B_perm = b[np.ix_(current_p, current_p)]
        # row_contrib[i] = sum_j a[i,j] * B_perm[i,j]
        row_contrib = a @ B_perm
        # col_contrib[j] = sum_i a[i,j] * B_perm[i,j]  
        col_contrib = a.T @ B_perm
        
        # For swapping positions i and j in current_p:
        # Let x = current_p[i], y = current_p[j]
        # Old terms involving i or j in the cost:
        # We need to carefully compute the delta.
        
        # The cost is sum_{i,j} a[i,j] * B_perm[i,j] where B_perm[i,j] = b[current_p[i], current_p[j]]
        # When we swap current_p[i] and current_p[j], B_perm changes in rows i,j and cols i,j.
        
        # Let's use a different approach: directly compute the change.
        # Old contribution from pairs involving i or j:
        # For facility i: sum_k a[i,k] * b[current_p[i], current_p[k]]
        # For facility j: sum_k a[j,k] * b[current_p[j], current_p[k]]
        # Plus cross terms a[i,j]*b[cp[i],cp[j]] and a[j,i]*b[cp[j],cp[i]]
        
        # New contribution after swap (x,y swapped):
        # For facility i: sum_k a[i,k] * b[current_p[k], current_p[k]] but with cp[i]<->cp[j]
        # This is complex. Let me use a simpler O(n) delta formula.
        
        # Alternative: compute delta by noting that only B_perm[i,:] and B_perm[:,i] and 
        # B_perm[j,:] and B_perm[:,j] change.
        
        no_improve = 0
        max_no_improve = 500
        best_cost = current_cost
        best_p = current_p.copy()
        
        while True:
            if max_time is not None and time.time() > max_time:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick random pair to swap
            i, j = np.random.choice(n, 2, replace=False)
            x = current_p[i]
            y = current_p[j]
            
            if x == y:
                continue
            
            # Compute delta for swapping positions i and j
            # Old: a[i,j]*b[x,y] + a[j,i]*b[y,x] + sum_{k!=i,j} a[i,k]*b[x,cp[k]] + sum_{k!=i,j} a[j,k]*b[y,cp[k]]
            #      + sum_{k!=i,j} a[k,i]*b[cp[k],x] + sum_{k!=i,j} a[k,j]*b[cp[k],y]
            # New: a[i,j]*b[y,x] + a[j,i]*b[x,y] + sum_{k!=i,j} a[i,k]*b[y,cp[k]] + sum_{k!=k,i,j} a[j,k]*b[x,cp[k]]
            #      + sum_{k!=i,j} a[k,i]*b[cp[k],y] + sum_{k!=i,j} a[k,j]*b[cp[k],x]
            
            # Delta = new - old
            # = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
            #   + sum_{k!=i,j} a[i,k]*(b[y,cp[k]]-b[x,cp[k]])
            #   + sum_{k!=i,j} a[j,k]*(b[x,cp[k]]-b[y,cp[k]])
            #   + sum_{k!=i,j} a[k,i]*(b[cp[k],y]-b[cp[k],x])
            #   + sum_{k!=i,j} a[k,j]*(b[cp[k],x]-b[cp[k],y])
            
            # Since b is symmetric: b[y,x] = b[x,y], so first two terms cancel!
            # Delta = sum_{k!=i,j} a[i,k]*(b[y,cp[k]]-b[x,cp[k]])
            #       + sum_{k!=i,j} a[j,k]*(b[x,cp[k]]-b[y,cp[k]])
            #       + sum_{k!=i,j} a[k,i]*(b[cp[k],y]-b[cp[k],x])
            #       + sum_{k!=i,j} a[k,j]*(b[cp[k],x]-b[cp[k],y])
            
            # We can vectorize this. Let's compute it efficiently.
            # For k != i,j:
            # term1[k] = a[i,k] * (b[y, current_p[k]] - b[x, current_p[k]])
            # term2[k] = a[j,k] * (b[x, current_p[k]] - b[y, current_p[k]])
            # term3[k] = a[k,i] * (b[current_p[k], y] - b[current_p[k], x])
            # term4[k] = a[k,j] * (b[current_p[k], x] - b[current_p[k], y])
            
            # Since b is symmetric, b[y, current_p[k]] = b[current_p[k], y], etc.
            # So term1[k] = a[i,k] * (b[current_p[k], y] - b[current_p[k], x])
            # term2[k] = a[j,k] * (b[current_p[k], x] - b[current_p[k], y])
            # term3[k] = a[k,i] * (b[current_p[k], y] - b[current_p[k], x])
            # term4[k] = a[k,j] * (b[current_p[k], x] - b[current_p[k], y])
            
            # Let d[k] = b[current_p[k], y] - b[current_p[k], x]
            # Then:
            # term1[k] = a[i,k] * d[k]
            # term2[k] = a[j,k] * (-d[k])
            # term3[k] = a[k,i] * d[k]
            # term4[k] = a[k,j] * (-d[k])
            
            # Sum = sum_{k!=i,j} d[k] * (a[i,k] - a[j,k] + a[k,i] - a[k,j])
            
            # Since a is symmetric: a[i,k] = a[k,i], a[j,k] = a[k,j]
            # Sum = sum_{k!=i,j} d[k] * (a[i,k] - a[j,k] + a[i,k] - a[j,k])
            #     = sum_{k!=i,j} d[k] * 2*(a[i,k] - a[j,k])
            
            # So delta = 2 * sum_{k!=i,j} (b[current_p[k], y] - b[current_p[k], x]) * (a[i,k] - a[j,k])
            
            # Vectorized computation:
            cp = current_p
            d = b[cp, y] - b[cp, x]  # shape (n,)
            coeff = a[i, :] - a[j, :]  # shape (n,)
            # Exclude k=i and k=j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            delta = 2.0 * np.sum(d[mask] * coeff[mask])
            
            if delta < 0:
                # Accept the swap
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                no_improve = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            else:
                no_improve += 1
        
        return best_p, best_cost

    # Step 1: Spectral initialization
    p_init = spectral_init()
    best_p, best_cost = local_search(p_init, max_time=start_time + 30.0)
    
    # Step 2: Multi-start with random perturbations
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:  # Reserve 5s for final polish
        # Create perturbed version of best solution
        p_perturb = best_p.copy()
        
        # Swap 5-15 random pairs
        n_swaps = rng.randint(5, 16)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_perturb[i], p_perturb[j] = p_perturb[j], p_perturb[i]
        
        # Run local search from perturbed solution
        p_local, cost_local = local_search(p_perturb, max_time=time.time() + 15.0)
        
        if cost_local < best_cost:
            best_cost = cost_local
            best_p = p_local
    
    # Step 3: Final polish with more iterations
    p_final, cost_final = local_search(best_p, max_time=deadline)
    
    if cost_final < best_cost:
        best_cost = cost_final
        best_p = p_final
    
    return best_p.tolist()
