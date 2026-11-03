# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if max_time is None:
            max_time = deadline
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Spectral Initialization ---
    # Use eigenvalue decomposition of the distance matrix B to get a structured initial solution
    B_sym = (b + b.T) / 2.0
    try:
        eigvals, eigvecs = np.linalg.eigh(B_sym)
        # Use the largest eigenvectors as coordinates
        # Sort by eigenvalue descending
        idx = np.argsort(-eigvals)
        eigvecs = eigvecs[:, idx]
        # Use first 2 eigenvectors for a 2D embedding
        coords = eigvecs[:, :2]
        # Sort facilities by their projected coordinates to create an initial ordering
        # Use a space-filling curve (Z-order) based on coordinates
        # Normalize coordinates to [0, 1]
        for dim in range(2):
            cmin = coords[:, dim].min()
            cmax = coords[:, dim].max()
            if cmax > cmin:
                coords[:, dim] = (coords[:, dim] - cmin) / (cmax - cmin)
        # Z-order sort
        z_order = np.zeros(n, dtype=np.int64)
        for dim in range(2):
            z_order += (coords[:, dim].astype(np.float32).view(np.uint32) >> (16 - dim * 8)).astype(np.int64) << (dim * 8)
        spectral_order = np.argsort(z_order, kind='stable')
        # Create permutation: facility i gets location spectral_order[i]
        p = spectral_order.astype(np.int64).copy()
    except Exception:
        # Fallback: rank matching
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind='stable')
        central_order = np.argsort(b_row_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        p = p.copy()

    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 20.0)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Iterated Hungarian Reassignment with Perturbation ---
    p = best_p.copy()
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 5

    while time.time() < deadline - 25.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(p)
        
        # Add small random perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + np.random.randn(n, n) * noise_scale
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break
            
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0
            
            # Polish after a good Hungarian step
            p, p_cost = two_opt_polish(p, max_time=min(deadline - 10.0, time.time() + 10.0))
            if p_cost < best_cost:
                best_p = p.copy()
                best_cost = p_cost
        else:
            no_improve_rounds += 1
            
            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = np.random.randint(5, 15)
                for _ in range(num_swaps):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                p, p_cost = two_opt_polish(p, max_time=min(deadline - 10.0, time.time() + 5.0))
                if p_cost < best_cost:
                    best_p = p.copy()
                    best_cost = p_cost
                no_improve_rounds = 0
                hungarian_count = 0

    # --- Phase 3: Final Polish ---
    p_polish = best_p.copy()
    p_polish, final_cost = two_opt_polish(p_polish, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_polish
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
