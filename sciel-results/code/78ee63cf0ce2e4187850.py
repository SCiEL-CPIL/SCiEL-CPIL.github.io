# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _init_permutation(a, b):
    """Rank-matching heuristic initialization."""
    n = len(a)
    # Sort facilities by total flow (descending) and locations by total distance (ascending)
    facility_rank = np.argsort(-a.sum(axis=1), kind="stable")
    location_rank = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[facility_rank] = location_rank
    return p


def _local_search(a, b, p, deadline, best_cost, best_p):
    """
    Best-improvement 2-opt local search with vectorized delta computation.
    Maintains B = b[p, p] for fast delta calculation.
    """
    n = len(p)
    B = b[np.ix_(p, p)]  # B[i,j] = b[p[i], p[j]]
    cur = int(np.sum(a * B))

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    while time.time() < deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1

        # For each i, compute delta of swapping i with all j > i
        # Delta(i,j) = sum_k (a[i,k] - a[j,k]) * (B[j,k] - B[i,k])
        #            + (a[i,i] - a[j,j]) * (B[j,j] - B[i,i])  [already included if we sum all k]
        # Actually, the full formula for symmetric a,b:
        # Delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (B[j,k] - B[i,k])
        #       + (a[i,i] - a[j,j]) * (B[j,j] - B[i,i])
        #       + (a[i,j] - a[j,i]) * (B[j,i] - B[i,j])  [zero if symmetric]
        #
        # If we define dA = a[i,:] - a[j,:] and dB = B[j,:] - B[i,:],
        # then sum_k dA[k]*dB[k] includes k=i and k=j terms.
        # dA[i]*dB[i] = (a[i,i]-a[j,i])*(B[i,i]-B[i,i]) = 0  [dB[i] = B[j,i]-B[i,i]]
        # Wait, dB[i] = B[j,i] - B[i,i], not zero.
        # Let me be more careful.
        #
        # B[i,k] = b[p[i], p[k]]
        # After swapping p[i] and p[j], new B[i,k] = b[p[j], p[k]] for k!=j, and B[j,k] = b[p[i],p[k]] for k!=i.
        #
        # The standard vectorized approach:
        # For fixed i, compute for all j:
        #   delta_j = (a[i,:] - a[j,:]) @ (B[j,:] - B[i,:])
        # This is O(n) per j, O(n^2) total per i, O(n^3) per pass.
        #
        # Better: precompute for each i, the vector D_i = a[i,:] - a[i,:] (no, that's zero)
        # 
        # Alternative: For fixed i, we want delta for all j.
        # delta(i,j) = sum_k (a[i,k] - a[j,k]) * (B[j,k] - B[i,k])
        #            = sum_k a[i,k]*B[j,k] - sum_k a[i,k]*B[i,k] - sum_k a[j,k]*B[j,k] + sum_k a[j,k]*B[i,k]
        #            = (a[i,:] @ B[j,:]) - C[i] - C[j] + (a[j,:] @ B[i,:])
        # where C[i] = sum_k a[i,k]*B[i,k]
        #
        # So if we maintain C, then for fixed i:
        #   delta(i,j) = a[i,:] @ B[j,:] - C[i] - C[j] + a[j,:] @ B[i,:]
        #
        # a[i,:] @ B[j,:] for all j is a[i,:] @ B.T (row i of a times B transposed)
        # a[j,:] @ B[i,:] for all j is B[i,:] @ a.T (row i of B times a transposed)
        #
        # So: delta(i, :) = a[i,:] @ B.T - C[i] - C + a[i,:] @ B.T ... wait
        # a[j,:] @ B[i,:] = (B[i,:] @ a.T)[j] = (a @ B[i,:].T)[j] = (a @ B[:,i])[j]
        #
        # Let me define:
        #   term1[j] = a[i,:] @ B[j,:]  for all j  -> this is (B.T @ a[i,:].T)[j] or a[i,:] @ B.T
        #   term2[j] = a[j,:] @ B[i,:]  for all j  -> this is (a @ B[i,:].T)[j] = (a @ B[:,i])[j]
        #
        # So delta(i, j) = term1[j] - C[i] - C[j] + term2[j]
        #
        # For fixed i:
        #   term1 = a[i,:] @ B.T   (vector of length n)
        #   term2 = a @ B[:, i]    (vector of length n)
        #   delta = term1 - C[i] - C + term2
        #
        # This is O(n^2) per i (two matrix-vector multiplies), O(n^3) per pass.
        # With n=100, that's 1M ops per pass, which is fast in NumPy.

        # Maintain C[i] = sum_k a[i,k] * B[i,k]
        C = (a * B).sum(axis=1)  # O(n^2)

        for i in range(n):
            # Compute delta for all j
            term1 = a[i, :] @ B.T  # a[i,:] @ B[j,:] for all j
            term2 = a @ B[:, i]    # a[j,:] @ B[i,:] for all j
            delta = term1 - C[i] - C + term2

            # Mask out j <= i (we only consider j > i to avoid duplicates)
            delta[:i+1] = 0  # We'll take min, so set non-candidates to 0

            j = int(np.argmin(delta))
            if j > i and delta[j] < best_delta:
                best_delta = delta[j]
                best_i = i
                best_j = j

        if best_i != -1:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            # Update B: swap rows i,j and columns i,j
            B[i, :], B[j, :] = B[j, :].copy(), B[i, :].copy()
            B[:, i], B[:, j] = B[:, j].copy(), B[:, i].copy()
            cur += int(best_delta)
            # Update C
            C[i] = (a[i, :] * B[i, :]).sum()
            C[j] = (a[j, :] * B[j, :]).sum()

            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
            improved = True

        if not improved:
            break

    return best_cost, best_p


def _perturb(p, n_swaps):
    """Apply random swaps to perturb the permutation."""
    n = len(p)
    p_perturbed = p.copy()
    for _ in range(n_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
    return p_perturbed


def solve(a, b):
    n = len(a)

    # 1. Initialization
    p = _init_permutation(a, b)

    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit

    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()

    # 2. Initial local search
    best_cost, best_p = _local_search(a, b, p.copy(), deadline, best_cost, best_p)

    # 3. Iterated Local Search
    T = 5000.0
    T_end = 1.0
    alpha = 0.9995

    while time.time() < deadline:
        # Perturb: 2-5 random swaps
        n_swaps = np.random.randint(2, 6)
        p_perturbed = _perturb(best_p, n_swaps)

        # Local search on perturbed solution
        cost_pert, p_pert = _local_search(a, b, p_perturbed, deadline, best_cost, best_p)

        # Update best
        if cost_pert < best_cost:
            best_cost = cost_pert
            best_p = p_pert.copy()
            T = max(T * 1.1, 5000.0)  # Reset temperature if improved
        else:
            # Accept with probability exp(-delta/T)
            delta = cost_pert - best_cost
            if np.random.rand() < np.exp(-delta / T):
                best_p = p_pert.copy()
                best_cost = cost_pert
            T = max(T * alpha, T_end)

    return best_p.tolist()
# EVOLVE-BLOCK-END
