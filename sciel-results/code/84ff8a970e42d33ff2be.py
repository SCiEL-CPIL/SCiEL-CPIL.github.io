"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Precompute for fast delta computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i a[i,:] dot b[p[i], :]
        total = 0
        for i in range(n):
            total += np.dot(a[i, :], b[p[i], :])
        return int(total)

    def compute_swap_delta(i, j, p):
        """
        Compute the change in cost when swapping locations of facilities i and j.
        p[i] and p[j] are the current locations.
        """
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0

        # Terms that change:
        # 1. Diagonal: a[i,i]*b[li,li] -> a[i,i]*b[lj,lj]
        #    a[j,j]*b[lj,lj] -> a[j,j]*b[li,li]
        # 2. Cross i-j: a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
        #    a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
        # 3. For all k not in {i,j}:
        #    a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]
        #    a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]
        #    a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]
        #    a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]

        delta = 0.0

        # Diagonal terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])

        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])

        # Terms with other facilities k (vectorized)
        # Create mask excluding i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]

        if len(ks) > 0:
            pk = p[ks]

            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))

        return int(delta)

    def local_search(p, max_time=None):
        """
        Steepest descent local search with 2-opt swaps.
        Returns the best permutation found.
        """
        if max_time is None:
            max_time = deadline

        cost = compute_cost(p)
        improved = True
        no_improve_count = 0

        while improved and time.time() < max_time:
            improved = False
            best_delta = 0
            best_pair = None

            # Find the best improving swap
            for i in range(n):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() >= max_time:
                        break
                    delta = compute_swap_delta(i, j, p)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)

            if best_pair is not None:
                i, j = best_pair
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

        return p, cost

    def perturb(p, strength=10):
        """
        Perturb the permutation by performing 'strength' random swaps.
        Used for ILS (Iterated Local Search).
        """
        p = p.copy()
        rng = np.random.RandomState(int(time.time() * 1000) % (2**32))
        for _ in range(strength):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    # Initial heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(p)

    # First local search from initial heuristic
    p_ls, cost_ls = local_search(p, deadline - 10)
    if cost_ls < best_cost:
        best_p = p_ls.copy()
        best_cost = cost_ls

    # Iterated Local Search (ILS)
    rng = np.random.RandomState(42)
    current_p = best_p.copy()
    current_cost = best_cost

    for iteration in range(100):
        if time.time() > deadline - 5:
            break

        # Perturb
        strength = 5 + int(10 * (1 - time.time() / deadline))
        strength = max(3, min(20, strength))
        perturbed_p = perturb(current_p, strength)

        # Local search from perturbed solution
        p_ls, cost_ls = local_search(perturbed_p, deadline - 5)

        # Accept if better
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
            current_p = p_ls.copy()
            current_cost = cost_ls
        else:
            # Sometimes accept worse solutions to escape local optima
            if rng.random() < 0.1:
                current_p = p_ls.copy()
                current_cost = cost_ls

    # Try a few different initial heuristics if time permits
    if time.time() < deadline - 30:
        # Inverse rank matching
        heavy2 = np.argsort(a.sum(axis=1), kind="stable")
        central2 = np.argsort(-b.sum(axis=1), kind="stable")
        p2 = np.empty(n, dtype=np.int64)
        p2[heavy2] = central2
        p2 = p2.astype(np.int64)
        p2_ls, cost2_ls = local_search(p2, deadline - 10)
        if cost2_ls < best_cost:
            best_p = p2_ls.copy()
            best_cost = cost2_ls

    if time.time() < deadline - 30:
        # Column-based matching
        heavy3 = np.argsort(-a.sum(axis=0), kind="stable")
        central3 = np.argsort(b.sum(axis=0), kind="stable")
        p3 = np.empty(n, dtype=np.int64)
        p3[heavy3] = central3
        p3 = p3.astype(np.int64)
        p3_ls, cost3_ls = local_search(p3, deadline - 10)
        if cost3_ls < best_cost:
            best_p = p3_ls.copy()
            best_cost = cost3_ls

    if time.time() < deadline - 30:
        # Random start
        p4 = rng.permutation(n).astype(np.int64)
        p4_ls, cost4_ls = local_search(p4, deadline - 10)
        if cost4_ls < best_cost:
            best_p = p4_ls.copy()
            best_cost = cost4_ls

    return best_p.tolist()
# EVOLVE-BLOCK-END
