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

    # Helper function for delta cost of swapping i and j
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]

        # Term 1: Change in cost due to rows i and j of A interacting with permuted B
        row_diff_a = a[i, :] - a[j, :]
        row_diff_b = b[v, p] - b[u, p]
        term1 = np.dot(row_diff_a, row_diff_b)

        # Term 2: Change in cost due to columns i and j of A interacting with permuted B
        col_diff_a = a[:, i] - a[:, j]
        col_diff_b = b[:, v][p] - b[:, u][p]
        term2 = np.dot(col_diff_a, col_diff_b)

        return term1 + term2

    # CTQA-inspired quantum perturbation
    def quantum_perturb(p, current_cost, temperature=1.0, num_steps=50):
        """
        Continuous-Time Quantum Annealing inspired perturbation.
        Uses a transverse field to allow tunneling through energy barriers.
        """
        p_work = p.copy()
        cost_work = current_cost
        
        # Precompute some quantities for efficiency
        B_p = b[p_work]  # B permuted by current p
        
        # Quantum annealing schedule: start with strong transverse field, decay to zero
        # s goes from 0 to 1
        for step in range(num_steps):
            if time.time() > deadline:
                break
                
            s = step / max(num_steps - 1, 1)
            # Transverse field strength: decreases as we anneal
            gamma = (1.0 - s) * 10.0  # Max field strength 10, decays to 0
            
            # At each step, try a swap but with quantum-inspired acceptance
            # The probability of accepting a worse move depends on the "quantum tunneling"
            # which is stronger when the field is large (early in annealing)
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = get_delta(i, j, p_work)
            
            if delta < 0:
                # Always accept improvements
                p_work[i], p_work[j] = p_work[j], p_work[i]
                cost_work += delta
                B_p = b[p_work]  # Update permuted B
            else:
                # Accept worse moves with probability based on quantum tunneling
                # P_accept = exp(-delta / (temperature * (1 + gamma)))
                # Higher gamma (stronger field) -> higher acceptance of worse moves
                effective_temp = temperature * (1.0 + gamma)
                if np.random.rand() < np.exp(-delta / max(effective_temp, 1e-10)):
                    p_work[i], p_work[j] = p_work[j], p_work[i]
                    cost_work += delta
                    B_p = b[p_work]
        
        return p_work, cost_work

    # Phase 1: Frank-Wolfe on the Birkhoff Polytope (Continuous Relaxation)
    fw_deadline = start_time + 40.0

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
            # Use quantum perturbation to escape local minimum
            improved = False
            for _ in range(5):
                p_q, cost_q = quantum_perturb(p, current_cost, temperature=1.0, num_steps=30)
                if cost_q < current_cost:
                    p = p_q
                    current_cost = cost_q
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                # If even quantum perturbation doesn't help, break to local search
                break

    # Phase 2: 2-Swap Local Search with O(n) delta evaluation
    p = best_p.copy()
    current_cost = best_cost

    no_improve_count = 0
    max_no_improve = 3000
    quantum_counter = 0

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Use quantum perturbation to escape local minimum
            quantum_counter += 1
            # Temperature increases slightly with each kick to explore more
            temp = 1.0 + 0.1 * min(quantum_counter, 10)
            p_q, cost_q = quantum_perturb(p, current_cost, temperature=temp, num_steps=40)
            
            if cost_q < current_cost:
                p = p_q
                current_cost = cost_q
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # If quantum perturbation didn't help, try a random kick
                i, j = np.random.choice(n, 2, replace=False)
                delta = get_delta(i, j, p)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            no_improve_count = 0
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
