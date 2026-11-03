import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def compute_cost(p):
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    def get_delta(p, i, j):
        li, lj = p[i], p[j]
        
        diff_a_row = a[i] - a[j]
        diff_b_col = b[lj, p] - b[li, p]
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, lj] - b[p, li]
        
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    def local_search(p, current_cost, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        return p, current_cost
                    
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        break
                if improved:
                    continue
                if time.time() > time_limit:
                    return p, current_cost
        return p, current_cost

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Run initial local search
    p, current_cost = local_search(p, current_cost, deadline - 20)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # Spectral Initialization
    # Construct a matrix that captures the interaction between flow and distance.
    # We want to find a permutation P such that sum a[i,j] b[P[i],P[j]] is minimized.
    # This is related to the alignment of the matrices.
    # A common spectral approach is to look at the eigenvectors of (A + A^T)/2 and (B + B^T)/2.
    # Since A and B are symmetric in QAP, we just use them.
    # We try to align the "principal components" of A and B.
    # Specifically, we can look at the matrix M = A \otimes B (Kronecker product) but that's huge.
    # Instead, we use the fact that if we define x as the vectorized permutation matrix,
    # the problem is quadratic.
    # A simpler spectral heuristic:
    # Compute the eigenvectors of A and B.
    # Sort facilities by the projection onto the top eigenvector of A.
    # Sort locations by the projection onto the top eigenvector of B.
    # Try different combinations of top eigenvectors (1st, 2nd, etc.) and signs.
    
    try:
        # Ensure symmetry for stability, though they should be symmetric
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        # Get top few eigenvectors
        # n=150, so full eigendecomposition is fast
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        # We want to minimize.
        # High flow should go to low distance?
        # Let's try aligning the principal directions.
        # For each pair of top eigenvectors (u from A, v from B),
        # sort facilities by |u[i]| or u[i] and locations by |v[j]| or v[j].
        
        # Let's try a few specific spectral initializations
        spectral_candidates = []
        
        # Top 5 eigenvectors
        for idx_a in range(5):
            u = evecs_a[:, idx_a]
            # Try sorting by u and -u
            for sign_u in [1, -1]:
                order_a = np.argsort(sign_u * u, kind='stable')
                for idx_b in range(5):
                    v = evecs_b[:, idx_b]
                    for sign_v in [1, -1]:
                        order_b = np.argsort(sign_v * v, kind='stable')
                        
                        p_spec = np.empty(n, dtype=np.int64)
                        # Assign facility at rank k in A-order to location at rank k in B-order?
                        # Or facility i gets location order_b[rank_of_i]?
                        # Let's do: facility i is placed at location order_b[rank_i]
                        # Where rank_i is the position of i in order_a
                        ranks_a = np.empty(n, dtype=np.int64)
                        ranks_a[order_a] = np.arange(n)
                        p_spec = order_b[ranks_a]
                        
                        spectral_candidates.append(p_spec)
        
        # Also try random projections (random subspace)
        for _ in range(10):
            u = np.random.randn(n)
            v = np.random.randn(n)
            order_a = np.argsort(u, kind='stable')
            order_b = np.argsort(v, kind='stable')
            ranks_a = np.empty(n, dtype=np.int64)
            ranks_a[order_a] = np.arange(n)
            p_spec = order_b[ranks_a]
            spectral_candidates.append(p_spec)
            
        # Evaluate and refine top candidates
        # We don't want to refine all of them if time is short.
        # Let's score them roughly first? No, compute_cost is fast.
        # But local_search is the expensive part.
        
        # Pick the best 5 spectral candidates
        scored_candidates = []
        for p_spec in spectral_candidates:
            c = compute_cost(p_spec)
            scored_candidates.append((c, p_spec))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        # Refine the top few
        time_left_for_refine = deadline - 5
        num_to_refine = min(5, len(scored_candidates))
        
        for i in range(num_to_refine):
            if time.time() > time_left_for_refine:
                break
            c_init, p_init = scored_candidates[i]
            # Allocate time per candidate
            time_per = (time_left_for_refine - time.time()) / (num_to_refine - i)
            time_limit_i = time.time() + time_per
            
            p_ref, c_ref = local_search(p_init.copy(), c_init, time_limit_i)
            
            if c_ref < best_cost:
                best_cost = c_ref
                best_p = p_ref.copy()
                
    except Exception:
        pass

    # Random Restarts
    # If we have time, perturb the best solution and try to improve it.
    while time.time() < deadline - 5:
        # Create a perturbed version of the best solution
        p_restart = best_p.copy()
        # Number of swaps: typically n/10 to n/5
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        # Calculate cost of perturbed solution
        current_cost_r = compute_cost(p_restart)
        
        # Run local search on the perturbed solution
        time_left = deadline - 5 - time.time()
        if time_left < 1:
            break
            
        p_r, cost_r = local_search(p_restart, current_cost_r, time.time() + time_left)
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()
            
    return best_p.tolist()
