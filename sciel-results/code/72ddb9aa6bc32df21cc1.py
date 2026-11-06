"""Rewritten program: conflict-graph + greedy + local search with multi-moves."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 150s limit

    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            gcd_table[i][j] = gcd(moduli[i], moduli[j])

    def find_valid_residue(m, moduli_list, residues):
        """Find a residue for m that doesn't conflict with existing assignment."""
        n = len(moduli_list)
        for r in range(m):
            ok = True
            for j in range(n):
                m2 = moduli_list[j]
                g = gcd_table[mod_idx[m]][mod_idx[m2]]
                if (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    def score_of(moduli_list):
        return sum(Fraction(1, m) for m in moduli_list)

    def greedy_initial(order):
        moduli_list = []
        residues = []
        for m in order:
            r = find_valid_residue(m, moduli_list, residues)
            if r is not None:
                moduli_list.append(m)
                residues.append(r)
        return moduli_list, residues

    best_moduli = []
    best_residues = []
    best_score = Fraction(0, 1)

    # --- Multiple greedy starts ---
    for attempt in range(10):
        if time.time() > deadline - 80:
            break

        if attempt == 0:
            order = list(range(20, 201))
        elif attempt == 1:
            order = list(range(20, 201, 2)) + list(range(21, 201, 2))
        elif attempt == 2:
            mult6 = [m for m in range(20, 201) if m % 6 == 0]
            mult4 = [m for m in range(20, 201) if m % 4 == 0 and m % 6 != 0]
            rest = [m for m in range(20, 201) if m % 6 != 0 and m % 4 != 0]
            order = mult6 + mult4 + rest
        elif attempt == 3:
            order = list(range(20, 201))
            random.shuffle(order)
        elif attempt == 4:
            order = list(range(200, 19, -1))
        elif attempt == 5:
            def is_prime(n):
                if n < 2:
                    return False
                for d in range(2, int(n**0.5) + 1):
                    if n % d == 0:
                        return False
                return True
            primes = [m for m in range(20, 201) if is_prime(m)]
            non_primes = [m for m in range(20, 201) if not is_prime(m)]
            order = non_primes + primes
        elif attempt == 6:
            def spf(n):
                for d in range(2, int(n**0.5) + 1):
                    if n % d == 0:
                        return d
                return n
            groups = {}
            for m in range(20, 201):
                p = spf(m)
                groups.setdefault(p, []).append(m)
            order = []
            for p in sorted(groups.keys()):
                order.extend(sorted(groups[p]))
        elif attempt == 7:
            small = list(range(20, 111))
            large = list(range(111, 201))
            order = []
            for i in range(max(len(small), len(large))):
                if i < len(small):
                    order.append(small[i])
                if i < len(large):
                    order.append(large[i])
        elif attempt == 8:
            # Multiples of 12 first
            mult12 = [m for m in range(20, 201) if m % 12 == 0]
            mult6 = [m for m in range(20, 201) if m % 6 == 0 and m % 12 != 0]
            mult4 = [m for m in range(20, 201) if m % 4 == 0 and m % 6 != 0]
            rest = [m for m in range(20, 201) if m % 12 != 0 and m % 6 != 0 and m % 4 != 0]
            order = mult12 + mult6 + mult4 + rest
        else:
            order = list(range(20, 201))
            random.shuffle(order)

        moduli_list, residues = greedy_initial(order)
        if moduli_list:
            s = score_of(moduli_list)
            if s > best_score:
                best_score = s
                best_moduli = moduli_list
                best_residues = residues

    # --- Local search: add, swap, multi-move ---
    if best_moduli and time.time() < deadline - 20:
        current_moduli = list(best_moduli)
        current_residues = list(best_residues)
        current_score = best_score

        improved = True
        while improved and time.time() < deadline - 15:
            improved = False

            # Try add
            for m in range(20, 201):
                if m in current_moduli:
                    continue
                r = find_valid_residue(m, current_moduli, current_residues)
                if r is not None:
                    s = current_score + Fraction(1, m)
                    if s > current_score:
                        current_moduli.append(m)
                        current_residues.append(r)
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_moduli = list(current_moduli)
                            best_residues = list(current_residues)
                        break

            if improved:
                continue

            # Try swap
            found_swap = False
            for i, m_remove in enumerate(current_moduli):
                if time.time() > deadline - 15:
                    break
                trial_moduli = [m for j, m in enumerate(current_moduli) if j != i]
                trial_residues = [r for j, r in enumerate(current_residues) if j != i]
                for m_add in range(20, 201):
                    if m_add in trial_moduli:
                        continue
                    r = find_valid_residue(m_add, trial_moduli, trial_residues)
                    if r is not None:
                        s = current_score - Fraction(1, m_remove) + Fraction(1, m_add)
                        if s > current_score:
                            current_moduli[i] = m_add
                            current_residues[i] = r
                            current_score = s
                            improved = True
                            found_swap = True
                            if s > best_score:
                                best_score = s
                                best_moduli = list(current_moduli)
                                best_residues = list(current_residues)
                            break
                if found_swap:
                    break

            if improved:
                continue

            # Try multi-move: remove 1, add 2
            if time.time() < deadline - 10:
                n = len(current_moduli)
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    m_remove = current_moduli[i]
                    trial_moduli = [m for j, m in enumerate(current_moduli) if j != i]
                    trial_residues = [r for j, r in enumerate(current_residues) if j != i]
                    base_score = current_score - Fraction(1, m_remove)

                    # Try adding pairs
                    for m1 in range(20, 201):
                        if m1 in trial_moduli:
                            continue
                        r1 = find_valid_residue(m1, trial_moduli, trial_residues)
                        if r1 is None:
                            continue
                        new_moduli = trial_moduli + [m1]
                        new_residues = trial_residues + [r1]
                        for m2 in range(m1 + 1, 201):
                            if m2 in new_moduli:
                                continue
                            r2 = find_valid_residue(m2, new_moduli, new_residues)
                            if r2 is None:
                                continue
                            s = base_score + Fraction(1, m1) + Fraction(1, m2)
                            if s > current_score:
                                current_moduli = new_moduli + [m2]
                                current_residues = new_residues + [r2]
                                current_score = s
                                improved = True
                                if s > best_score:
                                    best_score = s
                                    best_moduli = list(current_moduli)
                                    best_residues = list(current_residues)
                                break
                        if improved:
                            break
                    if improved:
                        break

    if not best_moduli:
        best_moduli = [20]
        best_residues = [0]
        best_score = Fraction(1, 20)

    result = [[best_moduli[i], best_residues[i]] for i in range(len(best_moduli))]
    return result
# EVOLVE-BLOCK-END
