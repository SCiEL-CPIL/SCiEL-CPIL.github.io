"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute masks: mask_for[(j, g, c)] = bitmask of residues r in [0, m_j) with r % g == c
    mask_for = {}
    for j in range(n_mod):
        mj = moduli[j]
        for g in range(1, mj + 1):
            for c in range(g):
                mask = 0
                for r in range(mj):
                    if r % g == c:
                        mask |= (1 << r)
                mask_for[(j, g, c)] = mask

    def greedy_init(order):
        avail = [(1 << moduli[i]) - 1 for i in range(n_mod)]
        chosen = []
        for idx in order:
            if avail[idx] == 0:
                continue
            mi = moduli[idx]
            best_r = -1
            best_cost = float('inf')
            for r in range(mi):
                if not (avail[idx] & (1 << r)):
                    continue
                cost = 0
                for j in range(idx + 1, n_mod):
                    if avail[j] == 0:
                        continue
                    g = gcd_mat[idx][j]
                    c = r % g
                    mask = mask_for[(j, g, c)]
                    cost += bin(avail[j] & mask).count('1')
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            if best_r >= 0:
                chosen.append((best_r, moduli[idx]))
                for j in range(idx + 1, n_mod):
                    g = gcd_mat[idx][j]
                    c = best_r % g
                    mask = mask_for[(j, g, c)]
                    avail[j] &= ~mask
        return chosen

    best_sol = []
    best_len = 0

    # Multiple orderings
    orderings = [
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: moduli[i]),
        sorted(range(n_mod), key=lambda i: -moduli[i]),
    ]

    def num_divisors(n):
        count = 0
        for d in range(1, int(n**0.5) + 1):
            if n % d == 0:
                count += 1
                if d != n // d:
                    count += 1
        return count

    orderings.append(sorted(range(n_mod), key=lambda i: num_divisors(moduli[i])))
    orderings.append(sorted(range(n_mod), key=lambda i: -num_divisors(moduli[i])))

    for order in orderings:
        if time.time() > deadline - 15.0:
            break
        sol = greedy_init(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol

    # Random orderings
    for trial in range(1000):
        if time.time() > deadline - 10.0:
            break
        random.seed(trial)
        order = list(range(n_mod))
        random.shuffle(order)
        sol = greedy_init(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol

    # Local improvement
    def local_improve(solution, time_limit):
        sol_moduli = set(m for _, m in solution)
        chosen = {m: r for r, m in solution}
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for m_add in moduli:
                if m_add in sol_moduli:
                    continue
                idx_add = moduli.index(m_add)
                avail = (1 << m_add) - 1
                for m_cur in sol_moduli:
                    idx_cur = moduli.index(m_cur)
                    g = gcd_mat[idx_cur][idx_add]
                    c = chosen[m_cur] % g
                    mask = mask_for[(idx_add, g, c)]
                    avail &= ~mask
                if avail == 0:
                    continue
                best_r = -1
                for r in range(m_add):
                    if avail & (1 << r):
                        best_r = r
                        break
                if best_r >= 0:
                    solution.append((best_r, m_add))
                    sol_moduli.add(m_add)
                    chosen[m_add] = best_r
                    improved = True
                    break
        return solution

    if time.time() < deadline - 5.0:
        best_sol = local_improve(best_sol, deadline - 5.0)
        best_len = len(best_sol)

    # More random trials with local improvement
    for trial in range(1000, 3000):
        if time.time() > deadline - 5.0:
            break
        random.seed(trial)
        order = list(range(n_mod))
        random.shuffle(order)
        sol = greedy_init(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_sol = sol
        if time.time() < deadline - 3.0:
            sol2 = local_improve(sol, deadline - 3.0)
            if len(sol2) > best_len:
                best_len = len(sol2)
                best_sol = sol2

    return [[r, m] for r, m in best_sol]
# EVOLVE-BLOCK-END
