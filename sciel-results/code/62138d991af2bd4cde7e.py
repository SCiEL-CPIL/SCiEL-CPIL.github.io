"""Rewritten program for the suite task; using a robust greedy with local search."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    MAX_MOD = 128

    # Precompute gcd for all pairs of moduli 2..128
    moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli)
    gcd_table = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid_solution(classes):
        """Verify pairwise disjointness."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            idx1 = m1 - 2
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                idx2 = m2 - 2
                g = gcd_table[idx1][idx2]
                if (r1 - r2) % g == 0:
                    return False
        return True

    def find_valid_residue(m, selected_classes, gcd_table, moduli):
        """
        Find a residue r for modulus m that is disjoint from all selected_classes.
        selected_classes: list of [r, m]
        """
        idx_m = m - 2
        for r in range(m):
            valid = True
            for r_prime, m_prime in selected_classes:
                idx_p = m_prime - 2
                g = gcd_table[idx_m][idx_p]
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_solve(order):
        """
        Run greedy algorithm for a given order of moduli.
        Returns list of [r, m].
        """
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected, gcd_table, moduli)
            if r is not None:
                selected.append([r, m])
        return selected

    def local_search(current_sol, unused_moduli, gcd_table, moduli, max_iter=100):
        """
        Try to add more moduli to current_sol by reassigning residues or adding new ones.
        """
        current = [list(c) for c in current_sol]
        used_mods = set(m for _, m in current)
        remaining = [m for m in moduli if m not in used_mods]

        improved = True
        iterations = 0
        while improved and time.time() < deadline and iterations < max_iter:
            improved = False
            iterations += 1

            # Try to add a new modulus from remaining
            remaining_sorted = sorted(remaining, reverse=True)

            added = False
            for m_new in remaining_sorted:
                r_new = find_valid_residue(m_new, current, gcd_table, moduli)
                if r_new is not None:
                    current.append([r_new, m_new])
                    used_mods.add(m_new)
                    remaining.remove(m_new)
                    improved = True
                    added = True
                    break

            if not added:
                if current and random.random() < 0.5:
                    idx_to_remove = random.randint(0, len(current) - 1)
                    r_rem, m_rem = current.pop(idx_to_remove)
                    used_mods.discard(m_rem)
                    remaining.append(m_rem)
                    for m_new in remaining:
                        r_new = find_valid_residue(m_new, current, gcd_table, moduli)
                        if r_new is not None:
                            current.append([r_new, m_new])
                            used_mods.add(m_new)
                            remaining.remove(m_new)
                            improved = True
                            break
                    if not improved:
                        current.append([r_rem, m_rem])
                        used_mods.add(m_rem)
                        remaining.remove(m_rem)

        return current

    best_solution = []
    best_size = 0

    # Strategy 1: Greedy with various orderings
    # 1. Descending (large moduli first)
    order1 = sorted(moduli, reverse=True)
    sol1 = greedy_solve(order1)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_solution = sol1

    # 2. Ascending
    order2 = sorted(moduli)
    sol2 = greedy_solve(order2)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_solution = sol2

    # 3. Random orderings
    num_randoms = 80
    for i in range(num_randoms):
        if time.time() > deadline:
            break
        random_order = moduli.copy()
        random.shuffle(random_order)
        sol_r = greedy_solve(random_order)
        if len(sol_r) > best_size:
            best_size = len(sol_r)
            best_solution = sol_r

    # Strategy 2: Local Search on the best greedy solution
    if best_solution and time.time() < deadline - 5.0:
        used_mods = set(m for _, m in best_solution)
        unused = [m for m in moduli if m not in used_mods]
        improved_sol = local_search(best_solution, unused, gcd_table, moduli, max_iter=300)
        if len(improved_sol) > best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol

    # Final verification
    if best_solution and not is_valid_solution(best_solution):
        # Fallback to a known valid small set if something went wrong
        fallback = []
        m = 2
        while m <= MAX_MOD:
            fallback.append([m // 2, m])
            m *= 2
        best_solution = fallback
        best_size = len(fallback)

    if not best_solution:
        best_solution = [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
