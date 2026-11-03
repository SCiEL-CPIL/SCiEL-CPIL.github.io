"""QAP tai256c solver: iterated local search with Hungarian re-optimization."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        # Build cost matrix: C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This is the marginal cost of assigning facility i to location j
        # given the current assignment of other facilities.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # We can compute this as: for each j, sum over k of a[i,k]*b[j,p[k]]
        # = a @ b[:, p]  ... wait, let me think more carefully.
        #
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # If we fix all assignments except facility i, then the cost contribution
        # involving facility i is:
        #   sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
        # = sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
        #
        # For facility i assigned to location j (instead of p[i]):
        #   marginal = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        #            - (terms where k=i should be excluded since p[i] changes)
        #
        # Actually for the Hungarian approach, we compute:
        # C[i, j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        # This is an approximation that ignores the self-interaction correction.
        # But it's good enough for a perturbation/re-optimization step.

        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        # First term: (a @ b[:, p])[i, j] ... no
        # sum_k a[i,k]*b[j,p[k]] = row i of (a @ b[:, p])? 
        # b[:, p] has columns rearranged: column k of b[:, p] is b[:, p[k]]
        # So (a @ b[:, p])[i, j] = sum_k a[i,k] * b[:,p][k,j] = sum_k a[i,k]*b[p[k], j]
        # That's the second term! sum_k a[i,k]*b[p[k], j]
        #
        # For the first term: sum_k a[i,k]*b[j, p[k]]
        # b[j, p[k]] is row j of b, column p[k]
        # = sum_k a[i,k] * b[j, p[k]]
        # Let B_p = b[np.ix_(np.arange(n), p)]  # B_p[j, k] = b[j, p[k]]
        # Then (B_p @ a.T)[j, i] = sum_k B_p[j,k] * a.T[k,i] = sum_k b[j,p[k]] * a[i,k]
        # So first term = (B_p @ a.T).T[i, j] = (a @ B_p.T)[i, j]? 
        # Let's just compute it directly with matrix multiply.

        # First term: M1[i,j] = sum_k a[i,k] * b[j, p[k]]
        # b[np.ix_(np.arange(n), p)] gives matrix where [j,k] = b[j, p[k]]
        # M1 = (a @ b[np.ix_(np.arange(n), p)].T) ... 
        # a @ X.T where X[j,k]=b[j,p[k]]: (a @ X.T)[i,j] = sum_k a[i,k]*X.T[k,j] = sum_k a[i,k]*X[j,k] = sum_k a[i,k]*b[j,p[k]] ✓
        
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        # Second term: M2[i,j] = sum_k a[k,i] * b[p[k], j]
        # b[np.ix_(p, np.arange(n))][k,j] = b[p[k], j]
        # M2 = (b[np.ix_(p, np.arange(n))] @ a)[i, j]? 
        # (Bp2 @ a)[i,j] where Bp2[k,j]=b[p[k],j]: sum_k Bp2[k,j]*a[j,i]... no
        # Let's think: M2[i,j] = sum_k a[k,i] * b[p[k], j]
        # = sum_k (a.T)[i,k] * b[p[k], j]
        # Bp2[k,j] = b[p[k], j]
        # (a.T @ Bp2)[i,j] = sum_k a.T[i,k] * Bp2[k,j] = sum_k a[k,i]*b[p[k],j] ✓
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = np.random.randint(2, 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 20:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
