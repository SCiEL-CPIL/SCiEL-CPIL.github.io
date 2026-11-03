"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def calc_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(p, i, j):
        """Compute cost change for swapping facilities at positions i and j."""
        pi, pj = p[i], p[j]
        # delta = sum_k a[i,k]*(b[pj,loc_k]-b[pi,loc_k]) + sum_k a[j,k]*(b[pi,loc_k]-b[pj,loc_k])
        #       = sum_k (a[i,k]-a[j,k])*(b[pj,loc_k]-b[pi,loc_k])
        # where loc_k = p[k], but we exclude i and j from the sum since they cancel
        dA = a[i, :] - a[j, :]
        dB = b[pj, :] - b[pi, :]
        # Zero out positions i and j to avoid double counting
        # Actually the formula is exact: the terms where k=i or k=j are included
        # Let's verify: when k=i, dB[p[i]]=b[pj,pi]-b[pi,pi], dA[i]=a[i,i]-a[j,i]
        # The full expansion includes all k, so no need to zero out
        return int(np.dot(dA, dB))

    def three_cycle_delta(p, i, j, k):
        """Compute cost change for cycling p[i]->p[j]->p[k]->p[i]."""
        pi, pj, pk = p[i], p[j], p[k]
        # New assignment: pos i gets pk, pos j gets pi, pos k gets pj
        # Old: pos i gets pi, pos j gets pj, pos k gets pk
        # Delta = sum over all pairs (x,y) of a[x,y] * (b[new[x],new[y]] - b[old[x],old[y]])
        # Only terms involving i, j, k change
        # Pairs: (i,i),(i,j),(i,k),(j,i),(j,j),(j,k),(k,i),(k,j),(k,k)
        # Old contributions:
        old_ii = a[i,i]*b[pi,pi]
        old_ij = a[i,j]*b[pi,pj]
        old_ik = a[i,k]*b[pi,pk]
        old_ji = a[j,i]*b[pj,pi]
        old_jj = a[j,j]*b[pj,pj]
        old_jk = a[j,k]*b[pj,pk]
        old_ki = a[k,i]*b[pk,pi]
        old_kj = a[k,j]*b[pk,pj]
        old_kk = a[k,k]*b[pk,pk]
        # New contributions (i gets pk, j gets pi, k gets pj):
        new_ii = a[i,i]*b[pk,pk]
        new_ij = a[i,j]*b[pk,pi]
        new_ik = a[i,k]*b[pk,pj]
        new_ji = a[j,i]*b[pi,pk]
        new_jj = a[j,j]*b[pi,pi]
        new_jk = a[j,k]*b[pi,pj]
        new_ki = a[k,i]*b[pj,pk]
        new_kj = a[k,j]*b[pj,pi]
        new_kk = a[k,k]*b[pj,pj]
        return (new_ii+new_ij+new_ik+new_ji+new_jj+new_jk+new_ki+new_kj+new_kk) - \
               (old_ii+old_ij+old_ik+old_ji+old_jj+old_jk+old_ki+old_kj+old_kk)

    def local_search_2opt(p, current_cost, max_passes=5):
        """2-opt local search with early termination."""
        for _pass in range(max_passes):
            if elapsed() > deadline - 5.0:
                break
            improved = False
            for i in range(n):
                if elapsed() > deadline - 5.0:
                    return p, current_cost, improved
                for j in range(i + 1, n):
                    d = swap_delta(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += d
                        improved = True
            if not improved:
                break
        return p, current_cost, improved

    def local_search_3opt(p, current_cost, max_passes=2):
        """3-opt local search (3-cycles)."""
        for _pass in range(max_passes):
            if elapsed() > deadline - 5.0:
                break
            improved = False
            for i in range(n):
                if elapsed() > deadline - 5.0:
                    return p, current_cost, improved
                for j in range(i + 1, n):
                    if elapsed() > deadline - 5.0:
                        return p, current_cost, improved
                    for k in range(j + 1, n):
                        d = three_cycle_delta(p, i, j, k)
                        if d < 0:
                            # Apply cycle: i gets p[k], j gets p[i], k gets p[j]
                            pi, pj, pk = p[i], p[j], p[k]
                            p[i], p[j], p[k] = pk, pi, pj
                            current_cost += d
                            improved = True
            if not improved:
                break
        return p, current_cost, improved

    def vnd_search(p, current_cost):
        """Variable Neighborhood Descent: alternate 2-opt and 3-opt."""
        for _round in range(6):
            if elapsed() > deadline - 5.0:
                break
            p, current_cost, imp2 = local_search_2opt(p, current_cost, max_passes=3)
            if elapsed() > deadline - 5.0:
                break
            p, current_cost, imp3 = local_search_3opt(p, current_cost, max_passes=1)
            if not imp2 and not imp3:
                break
        return p, current_cost

    def shake_3opt(p, current_cost, num_shakes=5):
        """Apply random 3-opt perturbations."""
        for _ in range(num_shakes):
            if elapsed() > deadline - 5.0:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            k = np.random.randint(n)
            # Ensure distinct
            if i == j or i == k or j == k:
                j = (j + 1) % n
                k = (k + 1) % n
                if i == j: j = (j + 1) % n
                if i == k or j == k: k = (k + 1) % n
                if i == j or i == k or j == k:
                    continue
            d = three_cycle_delta(p, i, j, k)
            pi, pj, pk = p[i], p[j], p[k]
            p[i], p[j], p[k] = pk, pi, pj
            current_cost += d
        return p, current_cost

    # Initialization: multiple heuristics
    best_p = None
    best_cost = float('inf')

    # Heuristic 1: rank-matching (heavy facilities to central locations)
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    c = calc_cost(p)
    best_p = p.copy()
    best_cost = c

    # Heuristic 2: reverse rank-matching
    heavy2 = np.argsort(-a.sum(axis=0), kind='stable')
    central2 = np.argsort(-b.sum(axis=1), kind='stable')
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    c2 = calc_cost(p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: random
    p3 = np.random.permutation(n)
    c3 = calc_cost(p3)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Main search loop: VND with shaking
    current_p = best_p.copy()
    current_cost = best_cost

    iteration = 0
    while elapsed() < deadline - 10.0:
        iteration += 1
        # VND local search
        current_p, current_cost = vnd_search(current_p, current_cost)

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()

        # Shake to escape local optimum
        current_p = best_p.copy()
        current_cost = best_cost
        current_p, current_cost = shake_3opt(current_p, current_cost, num_shakes=3 + iteration % 4)

        # Quick 2-opt to recover
        current_p, current_cost = local_search_2opt(current_p, current_cost, max_passes=2)

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()

    # Final polish: exhaustive 2-opt then 3-opt
    if elapsed() < deadline - 3.0:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = local_search_2opt(final_p, final_cost, max_passes=10)
        if elapsed() < deadline - 1.0:
            final_p, final_cost = local_search_3opt(final_p, final_cost, max_passes=1)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p

    return best_p.tolist()
# EVOLVE-BLOCK-END
