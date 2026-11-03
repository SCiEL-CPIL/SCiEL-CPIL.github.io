import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)

    # Spectral embedding initialization
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T

        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)

        # Take eigenvectors with largest absolute eigenvalues
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]  # (n, k_eig)
        V_B = eigvecs_B[:, idx_B]  # (n, k_eig)

        # Build assignment cost: align spectral signatures
        C_spectral = -V_A @ V_B.T  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    except Exception:
        # Fallback to rank-based initialization
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute A_sym if B is symmetric for faster C construction
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None

    # Phase 1: Frank-Wolfe on the Birkhoff Polytope (Continuous Relaxation)
    fw_deadline = start_time + 40.0

    while time.time() < fw_deadline:
        # Construct linear cost matrix C for the Frank-Wolfe step
        # C[i, k] represents the marginal cost of assigning facility i to location k
        # given the current assignment p.
        # C = A @ B_p.T + A.T @ B_p (general case)
        # If B is symmetric, C = (A + A.T) @ B_p

        B_p = b[p]  # Rows of B permuted by p. B_p[i, k] = B[p[i], k]

        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p

        # Solve linear assignment to get the next vertex of the Birkhoff polytope
        row_ind, col_ind = linear_sum_assignment(C)

        # New permutation
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        # Evaluate new cost
        new_cost = calc_cost(p_new)

        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement from the global linearization step.
            # We might be stuck in a basin where the linear approximation
            # doesn't lead to an immediate improvement, or we've converged.
            # Perform a few random swaps to perturb the solution before
            # attempting another linearization step.
            improved = False
            for _ in range(30):
                i, j = np.random.choice(n, 2, replace=False)
                p_swap = p.copy()
                p_swap[i], p_swap[j] = p_swap[j], p_swap[i]
                swap_cost = calc_cost(p_swap)
                if swap_cost < current_cost:
                    p = p_swap
                    current_cost = swap_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                # If even random swaps don't help immediately, we might be
                # in a very tight local optimum. Break to SA phase which
                # handles this better with temperature.
                break

    # Phase 2: 2-Swap Local Search with O(n) delta evaluation
    # This is significantly faster than O(n^2) cost calculation,
    # allowing for many more iterations to escape local minima.
    p = best_p.copy()
    current_cost = best_cost

    # Helper function for delta cost of swapping i and j
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]

        # Term 1: Change in cost due to rows i and j of A interacting with permuted B
        # Sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        row_diff_a = a[i, :] - a[j, :]
        row_diff_b = b[v, p] - b[u, p]
        term1 = np.dot(row_diff_a, row_diff_b)

        # Term 2: Change in cost due to columns i and j of A interacting with permuted B
        # Sum_k (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        col_diff_a = a[:, i] - a[:, j]
        col_diff_b = b[:, v][p] - b[:, u][p]
        term2 = np.dot(col_diff_a, col_diff_b)

        return term1 + term2

    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Escape local minimum with a random kick
            i, j = np.random.choice(n, 2, replace=False)
            delta = get_delta(i, j, p)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve_count = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            continue

        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue

        delta = get_delta(i, j, p)

        if delta < 0:
            # Accept improvement
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve_count = 0

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve_count += 1

    return best_p.tolist()
