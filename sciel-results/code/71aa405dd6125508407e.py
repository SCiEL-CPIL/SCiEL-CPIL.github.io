import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

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

    # Phase 1: Frank-Wolfe on the Birkhoff Polytope (Continuous Relaxation)
    fw_deadline = start_time + 40.0

    # Precompute A_sym if B is symmetric for faster C construction
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None

    while time.time() < fw_deadline:
        # Construct linear cost matrix C for the Frank-Wolfe step
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
            # Perform a few random swaps to perturb the solution
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
                # If even random swaps don't help immediately, break to SA phase
                break

    # Phase 2: Simulated Annealing (Fine-grained search)
    p = best_p.copy()
    current_cost = best_cost

    # SA Parameters
    T = 1000.0
    T_end = 1.0
    alpha = 0.99998  # Very slow cooling to maintain exploration

    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue

        p[i], p[j] = p[j], p[i]
        new_cost = calc_cost(p)
        delta = new_cost - current_cost

        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            if delta > T * 700:
                accept = False
            else:
                accept = (np.random.rand() < np.exp(-delta / T))

        if accept:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]

        T *= alpha
        if T < T_end:
            T = T_end

    return best_p.tolist()
