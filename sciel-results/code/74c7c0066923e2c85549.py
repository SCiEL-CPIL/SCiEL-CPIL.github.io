import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Improved Initial Assignment using Eigenvector Projection
    # Symmetrize matrices for eigendecomposition (QAP matrices are often not symmetric, but the "shape" is captured by symmetric parts or Laplacians)
    # Using the symmetric part is a common heuristic for QAP initialization.
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Compute top K eigenvectors
        K = 20
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Take the largest K eigenvectors (eigh returns in ascending order)
        # We want the ones with largest magnitude (eigenvalues) to capture the main structure
        # Note: For QAP, we often want to align the "high flow" with "short distance".
        # The eigenvectors represent the principal axes of the graph.
        
        # Select top K indices
        idx_a = np.argsort(eigvals_a)[::-1][:K]
        idx_b = np.argsort(eigvals_b)[::-1][:K]
        
        # Project facilities (rows of a_sym) and locations (rows of b_sym)
        # Actually, the eigenvectors ARE the coordinates.
        # Facility i is at eigvecs_a[:, i]
        # Location j is at eigvecs_b[:, j]
        
        # We want to minimize sum A[i,j] B[p[i], p[j]].
        # If we approximate A ~ V_a Lambda_a V_a^T and B ~ V_b Lambda_b V_b^T
        # Then cost ~ trace(V_a^T P^T V_b Lambda_b V_b^T P V_a Lambda_a V_a^T) ... this is complex.
        # Simpler heuristic: Minimize the distance between the projected points.
        # If facility i is "close" to facility j in A-space, they should be "close" in B-space.
        
        # Let's use the projected coordinates to compute a distance matrix for initialization.
        # Coordinates for facilities: eigvecs_a[:, idx_a].T  (shape K x n)
        # Coordinates for locations: eigvecs_b[:, idx_b].T  (shape K x n)
        
        fac_coords = eigvecs_a[:, idx_a].T
        loc_coords = eigvecs_b[:, idx_b].T
        
        # Compute pairwise squared distances
        # dist_fac[i, j] = || fac_coords[i] - fac_coords[j] ||^2
        # This is O(n^2 * K)
        # We can compute this efficiently
        fac_norms = np.sum(fac_coords**2, axis=1)[:, None]
        loc_norms = np.sum(loc_coords**2, axis=1)[:, None]
        
        # dist[i,j] = fac_norms[i] + fac_norms[j] - 2 * fac_coords[i] @ fac_coords[j]
        # We want to match high A-flow with low B-distance?
        # The term is A[i,j] * B[p[i], p[j]].
        # If A[i,j] is large, we want B[p[i], p[j]] to be small.
        # So we want to pair large A entries with small B entries.
        
        # Let's create a cost matrix for the initial Hungarian assignment.
        # We can approximate the "effective cost" of pairing i and j based on their ranks or projections.
        # A simpler and robust method:
        # Use the row sums of A and B as in the original, but refine with eigenvector alignment.
        
        # Let's stick to a robust refinement of the rank matching or use the eigenvectors directly to compute a "similarity" matrix.
        # Similarity between facility i and j: S_a[i,j] = fac_coords[i] @ fac_coords[j]
        # Distance between location i and j: D_b[i,j] = || loc_coords[i] - loc_coords[j] ||^2
        
        # We want to minimize sum A[i,j] B[p[i], p[j]].
        # This is hard to reduce to a linear assignment directly.
        # However, we can use the eigenvectors to define a "center of mass" or use a greedy construction.
        
        # Let's try a different initialization:
        # Start with rank matching, then do a few steps of the main loop immediately?
        # Or, use the eigenvectors to order facilities and locations.
        
        # Let's use the first principal component to sort.
        # This is a heuristic.
        
        # Actually, the current rank matching is okay. Let's focus on the Polish.
        # If we don't have a strong init, the Hungarian loop will do the work.
        
        # Let's just use the rank matching as before, it's safe.
        pass
    except Exception:
        pass

    # Fallback / Primary Init: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost_full(p):
        return np.sum(a * b[p][:, p])
    
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    def compute_dynamic_cost(p):
        b_jp = b[:, p]
        b_pj = b[p, :]
        C_first = a @ b_jp.T
        C_second = a.T @ b_pj
        return C_first + C_second

    # Main loop
    iteration = 0
    T = 1000.0
    min_T = 1.0
    
    while time.time() < deadline - 10.0: # Leave 10s for polish
        C = compute_dynamic_cost(p)
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        delta = new_cost - current_cost
        
        if delta < 0 or (T > min_T and np.random.rand() < np.exp(-delta / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            p = new_p.copy()
            current_cost = new_cost
            T = max(min_T, T * 0.95)
        else:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                swap_cost = calc_cost_full(p)
                swap_delta = swap_cost - current_cost
                if swap_delta < 0 or (T > min_T and np.random.rand() < np.exp(-swap_delta / T)):
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_p = p.copy()
                    current_cost = swap_cost
                    T = max(min_T, T * 0.95)
                else:
                    p[i], p[j] = p[j], p[i]
                    T = max(min_T, T * 0.99)
        iteration += 1

    # Polish Phase: 2-Swap Local Search
    # We use the best solution found so far
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute helpers for fast delta calculation?
    # Delta for swapping i and j (u=p[i], v=p[j]):
    # Remove: A[i,i]B[u,u] + A[i,j]B[u,v] + A[j,i]B[v,u] + A[j,j]B[v,v]
    #         + sum_{k!=i,j} (A[i,k]B[u,p[k]] + A[k,i]B[p[k],u] + A[j,k]B[v,p[k]] + A[k,j]B[p[k],v])
    # Add:    A[i,i]B[v,v] + A[i,j]B[v,u] + A[j,i]B[u,v] + A[j,j]B[u,u]
    #         + sum_{k!=i,j} (A[i,k]B[v,p[k]] + A[k,i]B[p[k],v] + A[j,k]B[u,p[k]] + A[k,j]B[p[k],u])
    
    # Vectorized check for all pairs is O(n^3) per pass.
    # n=256, n^3 ~ 16M.
    # We can do a few passes.
    
    # Optimization: We can compute the "interaction" of each facility i with the rest of the system at its current location.
    # Let R_i = sum_{k!=i} (A[i,k]B[p[i],p[k]] + A[k,i]B[p[k],p[i]])
    # This is the "cost" of i in the current configuration, excluding self and pair interactions with other specific j.
    # Actually, let's just do a simple hill climbing with a few iterations.
    
    # To make it fast, we'll evaluate a random subset of swaps or do a full pass if time permits.
    # Given the 10s budget, we can afford ~5-10 full O(n^3) passes if optimized in NumPy.
    
    # Let's try to vectorize the delta calculation for a single pass.
    # This is tricky in pure NumPy without loops over i,j.
    # We will do a "First Improvement" strategy with a limited number of attempts per pass.
    
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        # Randomly sample pairs to check
        # Or do a full scan if n is small enough?
        # Let's do a full scan but break early if no improvement found in first 1000 checks?
        # No, let's just do a few random restarts of local search.
        
        # Actually, let's just do a few iterations of "Best Improvement" 2-swap.
        # To avoid O(n^3) Python loops, we rely on the fact that n=256 is small.
        # But 16M operations in a Python loop is too slow.
        # We must vectorize or use a smarter heuristic.
        
        # Heuristic: Check swaps involving the "most expensive" facilities.
        # Or just do 100 random swaps and keep the best ones.
        
        # Let's do 500 random 2-swaps.
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            u = p[i]
            v = p[j]
            
            # Calculate delta
            # Terms involving i and j
            # Old: A[i,i]B[u,u] + A[i,j]B[u,v] + A[j,i]B[v,u] + A[j,j]B[v,v]
            old_self = a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
            # New: A[i,i]B[v,v] + A[i,j]B[v,u] + A[j,i]B[u,v] + A[j,j]B[u,u]
            new_self = a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]
            
            # Interactions with others
            # sum_{k!=i,j} A[i,k]B[u,p[k]] + A[k,i]B[p[k],u] + A[j,k]B[v,p[k]] + A[k,j]B[p[k],v]
            # vs
            # sum_{k!=i,j} A[i,k]B[v,p[k]] + A[k,i]B[p[k],v] + A[j,k]B[u,p[k]] + A[k,j]B[p[k],u]
            
            # We can compute the sum over all k and subtract the i,j terms if they were included,
            # but it's easier to just compute the sum over all k and adjust.
            # Let S_i_u = sum_k A[i,k]B[u,p[k]]
            # Let S_i_v = sum_k A[i,k]B[v,p[k]]
            # Let T_i_u = sum_k A[k,i]B[p[k],u]
            # Let T_i_v = sum_k A[k,i]B[p[k],v]
            
            # We can precompute these for all i and u,v? No, too much memory/time.
            # We compute them on the fly for the specific i and j.
            
            # S_i_u = a[i,:] @ b[u, p]
            # S_i_v = a[i,:] @ b[v, p]
            # T_i_u = a[:,i] @ b[p, u]
            # T_i_v = a[:,i] @ b[p, v]
            
            # Note: b[u, p] is a vector of length n.
            # b[p, u] is a vector of length n.
            
            S_i_u = np.dot(a[i, :], b[u, p])
            S_i_v = np.dot(a[i, :], b[v, p])
            T_i_u = np.dot(a[:, i], b[p, u])
            T_i_v = np.dot(a[:, i], b[p, v])
            
            S_j_u = np.dot(a[j, :], b[u, p])
            S_j_v = np.dot(a[j, :], b[v, p])
            T_j_u = np.dot(a[:, j], b[p, u])
            T_j_v = np.dot(a[:, j], b[p, v])
            
            # The sums S and T include k=i and k=j.
            # We need to exclude k=i and k=j from the "others" part.
            # Old Others = (S_i_u - A[i,i]B[u,p[i]] - A[i,j]B[u,p[j]]) + (T_i_u - A[i,i]B[p[i],u] - A[j,i]B[p[j],u])
            #             + (S_j_u - A[j,i]B[u,p[i]] - A[j,j]B[u,p[j]]) + (T_j_u - A[i,j]B[p[i],u] - A[j,j]B[p[j],u])
            # This is getting complicated.
            
            # Simpler: Just compute the full cost change by summing over all k, and correcting for double counting or self terms.
            # Actually, the formula:
            # Delta = (New Self) - (Old Self) + (New Others) - (Old Others)
            
            # New Others = sum_{k!=i,j} (A[i,k]B[v,p[k]] + A[k,i]B[p[k],v] + A[j,k]B[u,p[k]] + A[k,j]B[p[k],u])
            # Old Others = sum_{k!=i,j} (A[i,k]B[u,p[k]] + A[k,i]B[p[k],u] + A[j,k]B[v,p[k]] + A[k,j]B[p[k],v])
            
            # Let's compute the sums over ALL k first.
            # Sum_i_v_all = S_i_v (includes k=i, k=j)
            # Sum_i_u_all = S_i_u
            # Sum_T_i_v_all = T_i_v
            # Sum_T_i_u_all = T_i_u
            # Same for j.
            
            # The terms for k=i in S_i_v is A[i,i]B[v,p[i]] = A[i,i]B[v,u] (since p[i]=u)
            # The terms for k=j in S_i_v is A[i,j]B[v,p[j]] = A[i,j]B[v,v] (since p[j]=v)
            
            # This manual correction is error-prone.
            # Given the time, let's just use `calc_cost_full` for the polish if we limit the number of swaps.
            # 500 swaps * O(n^2) = 500 * 65k = 32M ops. This is fine in 10s.
            
            p[i], p[j] = p[j], p[i]
            new_swap_cost = calc_cost_full(p)
            delta = new_swap_cost - current_cost
            
            if delta < 0:
                current_cost = new_swap_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
            else:
                p[i], p[j] = p[j], p[i]
                
        if not improved:
            break

    return best_p.tolist()
