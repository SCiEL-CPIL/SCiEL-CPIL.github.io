"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - aj_i) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (ai_j - aj_j) * (bu_v - bv_v)
        
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 300
        iter_count = 0
        
        while improved and time.time() < deadline - 3.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n - 1):
                if time.time() >= deadline - 3.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 3.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    def spectral_init(a, b, k=10, seed=42):
        """
        Spectral initialization for QAP.
        
        The idea: create a matrix M that captures the interaction between
        flow and distance structures, then use its top eigenvectors to
        generate diverse initial assignments.
        
        For QAP, we want to align high-flow facility pairs with low-distance
        location pairs. A spectral approach can capture this global alignment.
        """
        rng = np.random.RandomState(seed)
        
        # Create a combined matrix that reflects the QAP structure
        # Normalize a and b to have zero mean and unit variance for better conditioning
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        # Scale to prevent numerical issues
        a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
        b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
        
        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale
        
        # Build a matrix whose eigenvectors encode alignment between a and b
        # The QAP cost can be written as sum_{i,j} a_ij * b_{p(i),p(j)}
        # We want p to minimize this, so we want to align large a_ij with small b_p(i),p(j)
        
        # One approach: use the Hadamard product structure
        # M = a_norm ⊙ b_norm doesn't directly work for spectral methods
        
        # Better approach: use the fact that we can think of this as
        # finding a permutation that minimizes trace(A P B P^T)
        # where P is the permutation matrix.
        
        # Spectral relaxation: replace P with a continuous variable
        # and solve an eigenvalue problem.
        
        # For initialization, we'll use a simpler heuristic:
        # 1. Compute the "importance" of each facility from a
        # 2. Compute the "centrality" of each location from b
        # 3. Use eigenvectors to create multiple diverse initial solutions
        
        # Method 1: Rank-based (baseline)
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        
        p_rank_rev = np.empty(n, dtype=np.int64)
        p_rank_rev[heavy] = central[::-1]
        
        # Method 2: Spectral using top eigenvectors of a combined matrix
        # Create a matrix that captures the joint structure
        # C = a_norm + b_norm (symmetrized)
        C = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
        C = 0.5 * (C + C.T)  # Ensure symmetry
        
        # Get top k eigenvectors
        try:
            if time.time() < deadline - 30.0:
                eigenvalues, eigenvectors = eigh(C, subset_by_index=[n-k, n-1])
                
                # Use each eigenvector to create a permutation
                # Sort facilities by their coordinate in the eigenvector
                # and assign to locations sorted by their coordinate
                
                p_spectral = []
                for idx in range(k):
                    eigvec = eigenvectors[:, idx]
                    
                    # Sort facilities by their value in this eigenvector
                    facility_order = np.argsort(eigvec, kind="stable")
                    
                    # Sort locations by their value in this eigenvector (reversed for minimization)
                    location_order = np.argsort(-eigvec, kind="stable")
                    
                    # Create permutation: facility at position facility_order[i] goes to location location_order[i]
                    p = np.empty(n, dtype=np.int64)
                    for i in range(n):
                        p[facility_order[i]] = location_order[i]
                    
                    p_spectral.append(p)
        except Exception:
            p_spectral = []
        
        # Method 3: Random with structure
        p_random = rng.permutation(n)
        
        # Combine all initializations
        inits = [p_rank, p_rank_rev, p_random] + p_spectral[:min(3, len(p_spectral))]
        
        return inits

    # Generate diverse initializations
    inits = spectral_init(a, b, k=8, seed=12345)
    
    # Add a few more random initializations for diversity
    rng = np.random.RandomState(67890)
    for _ in range(2):
        inits.append(rng.permutation(n))
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    # Iterated Local Search (ILS) to escape local minima
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
    
    # Adaptive perturbation: start small, increase if stuck
    base_perturb = 5
    max_perturb = 20
    no_improve_count = 0
    
    while time.time() < deadline - 8.0:
        # Perturb the best solution
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 5) * 2
        perturb_size = min(perturb_size, max_perturb)
        
        num_swaps = np.random.randint(max(2, perturb_size // 2), perturb_size + 1)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If we've tried many times without improvement, increase perturbation
        if no_improve_count > 10:
            no_improve_count = 0  # Reset to continue searching with current perturbation level
    
    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
