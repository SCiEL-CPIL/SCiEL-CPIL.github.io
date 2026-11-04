import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)

    # Phase 0: Hungarian initialization using flow/distance centrality
    # Facility centrality: total flow from/to facility i
    # Location centrality: total distance from/to location j
    a_row_sums = a.sum(axis=1)  # outflow from facility i
    a_col_sums = a.sum(axis=0)  # inflow to facility i
    fac_centrality = a_row_sums + a_col_sums  # total flow involving facility i

    b_row_sums = b.sum(axis=1)  # total distance from location j
    b_col_sums = b.sum(axis=0)  # total distance to location j
    loc_centrality = b_row_sums + b_col_sums  # total distance involving location j

    # Cost matrix: negative product of centralities (we want to minimize, so minimize negative = maximize product)
    # High-flow facilities should go to high-distance locations? No - we want to minimize sum a[i,j]*b[p[i],p[j]]
    # So high-flow pairs should go to low-distance locations.
    # Cost[i,j] = fac_centrality[i] * loc_centrality[j] -- we want to MINIMIZE this
    C_init = np.outer(fac_centrality, loc_centrality)
    row_ind, col_ind = linear_sum_assignment(C_init)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    # Also try spectral initialization and pick the better one
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T
        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]
        V_A = eigvecs_A[:, idx_A]
        V_B = eigvecs_B[:, idx_B]
        C_spectral = -V_A @ V_B.T
        row_ind_s, col_ind_s = linear_sum_assignment(C_spectral)
        p_spec = np.empty(n, dtype=np.int64)
        p_spec[row_ind_s] = col_ind_s
    except Exception:
        p_spec = p.copy()

    # Compute cost for both and pick better
    def calc_cost(p_arr):
        b_perm = b[p_arr][:, p_arr]
        return np.sum(a * b_perm)

    cost_central = calc_cost(p)
    cost_spec = calc_cost(p_spec)
    if cost_spec < cost_central:
        p = p_spec
        current_cost = cost_spec
    else:
        current_cost = cost_central

    best_p = p.copy()
    best_cost = current_cost

    # Precompute for fast swap delta: delta(i,j) = sum_k a[i,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + sum_k a[k,i]*(b[p[k],p[i]] - b[p[k],p[j]])
    # Actually for QAP with symmetric A and B:
    # cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
    # Swapping p[i] and p[j]: the change involves terms where i or j is involved
    # Let's precompute row/col contributions

    # For fast evaluation, precompute:
    # row_contrib[i] = sum_k a[i,k] * b[p[i], p[k]]  (outgoing from i)
    # col_contrib[i] = sum_k a[k,i] * b[p[k], p[i]]  (incoming to i)
    # total = sum_i (row_contrib[i] + col_contrib[i]) but this double counts diagonal
    # Actually cost = sum_{i,j} a[i,j]*b[p[i],p[j]], so let's just compute the delta directly

    def compute_swap_delta(p_arr, i, j):
        """Compute cost change when swapping positions i and j in permutation p_arr."""
        pi = p_arr[i]
        pj = p_arr[j]
        # Terms affected: those involving facility i or j, or location pi or pj
        # delta = sum_k [a[i,k]*(b[pi,p[k]] -> b[pj,p[k]]) + a[j,k]*(b[pj,p[k]] -> b[pi,p[k]])]
        #       + sum_k [a[k,i]*(b[p[k],pi] -> b[p[k],pj]) + a[k,j]*(b[p[k],pj] -> b[p[k],pi])]
        #       - cross terms where k=i or k=j need care
        # Simpler: compute the difference directly
        # Old terms involving i or j:
        # For facility i: sum_k a[i,k] * b[pi, p[k]]
        # For facility j: sum_k a[j,k] * b[pj, p[k]]
        # For location pi (as second index): sum_k a[k,i] * b[p[k], pi]
        # For location pj (as second index): sum_k a[k,j] * b[p[k], pj]
        # New terms after swap:
        # For facility i: sum_k a[i,k] * b[pj, p[k]]
        # For facility j: sum_k a[j,k] * b[pi, p[k]]
        # For location pj: sum_k a[k,i] * b[p[k], pj]
        # For location pi: sum_k a[k,j] * b[p[k], pi]
        
        # Vectorized computation
        # b_pi = b[pi, :] is the row of B for location pi
        # b_pj = b[pj, :] is the row of B for location pj
        
        # delta from facility i's perspective: sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        # = sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        # = a[i, :] @ (b[pj, p_arr] - b[pi, p_arr])
        
        # delta from facility j's perspective: sum_k a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
        # = a[j, :] @ (b[pi, p_arr] - b[pj, p_arr])
        
        # delta from location pi's perspective (as column): sum_k a[k,i] * (b[p[k], pj] - b[p[k], pi])
        # = a[:, i] @ (b[p_arr, pj] - b[p_arr, pi])
        
        # delta from location pj's perspective (as column): sum_k a[k,j] * (b[p[k], pi] - b[p[k], pj])
        # = a[:, j] @ (b[p_arr, pi] - b[p_arr, pj])
        
        b_pi_row = b[pi, p_arr]  # (n,)
        b_pj_row = b[pj, p_arr]  # (n,)
        b_col_pi = b[p_arr, pi]  # (n,)
        b_col_pj = b[p_arr, pj]  # (n,)
        
        delta_i = a[i, :] @ (b_pj_row - b_pi_row)
        delta_j = a[j, :] @ (b_pi_row - b_pj_row)
        delta_loc_pi = a[:, i] @ (b_col_pj - b_col_pi)
        delta_loc_pj = a[:, j] @ (b_col_pi - b_col_pj)
        
        return delta_i + delta_j + delta_loc_pi + delta_loc_pj

    # Phase 1: Fast local search with vectorized swap deltas
    local_deadline = start_time + 30.0
    improved = True
    while time.time() < local_deadline and improved:
        improved = False
        # Try all pairs in random order until no improvement
        order = np.random.permutation(n)
        for i in order:
            if time.time() >= local_deadline:
                break
            for j in order[i+1:]:
                if time.time() >= local_deadline:
                    break
                delta = compute_swap_delta(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    break  # Restart from beginning after improvement

    # Phase 2: Frank-Wolfe for global moves
    fw_deadline = start_time + 60.0
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None

    while time.time() < fw_deadline:
        B_p = b[p]
        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p

        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        new_cost = calc_cost(p_new)
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Try a few random swaps to escape
            esc = False
            for _ in range(20):
                i, j = np.random.choice(n, 2, replace=False)
                delta = compute_swap_delta(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    esc = True
                    break
            if not esc:
                break

    # Phase 3: Simulated Annealing with fast delta
    p = best_p.copy()
    current_cost = best_cost

    T = 1000.0
    T_end = 1.0
    # Estimate steps per second to set alpha appropriately
    # We have about 50 seconds left
    remaining_time = deadline - time.time()
    # Target ~500k steps, so alpha = (T_end/T)^(1/steps)
    target_steps = 500000
    alpha = (T_end / T) ** (1.0 / target_steps) if target_steps > 0 else 0.999

    step_count = 0
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue

        delta = compute_swap_delta(p, i, j)

        if delta < 0:
            accept = True
        else:
            if delta > T * 700:
                accept = False
            else:
                accept = (np.random.rand() < np.exp(-delta / T))

        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            pass  # No change needed since we didn't modify p

        T *= alpha
        if T < T_end:
            T = T_end
        step_count += 1

        # Check time every 1000 steps
        if step_count % 1000 == 0 and time.time() >= deadline:
            break

    return best_p.tolist()
