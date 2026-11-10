import time
import random
from math import gcd

def construct():
    start_time = time.time()
    max_time = 50.0  # Safety margin before 60s hard limit

    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_moduli = len(moduli)

    # Precompute gcd table
    gcd_table = [[0] * (max_modulus + 1) for _ in range(max_modulus + 1)]
    for i in range(2, max_modulus + 1):
        for j in range(2, max_modulus + 1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute for each modulus m, the set of residues r in [0,m)
    # For each pair (m, m_j) with gcd g, the forbidden residues are those where r % g == r_j % g
    # We can precompute the "conflict mask" for each pair

    def find_valid_residues_fast(selected, m):
        """Find all valid residues for m given selected classes using precomputed conflicts."""
        if not selected:
            return list(range(m))

        # For each residue r in [0,m), check if it conflicts with any selected
        # A conflict exists if gcd(m, m_j) divides (r - r_j)
        valid = []
        for r in range(m):
            ok = True
            for r_j, m_j in selected:
                g = gcd_table[m][m_j]
                if (r - r_j) % g == 0:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    def greedy_with_order(order, randomize=False):
        """Greedy: process moduli in order, add if possible."""
        selected = []
        selected_moduli = set()
        for m in order:
            if time.time() > start_time + max_time:
                break
            if m in selected_moduli:
                continue
            valid = find_valid_residues_fast(selected, m)
            if valid:
                r = random.choice(valid) if randomize else valid[0]
                selected.append((r, m))
                selected_moduli.add(m)
        return selected

    def try_add_missing(selected, time_limit):
        """Try to add all missing moduli in various orders."""
        best = list(selected)
        selected_moduli = set(m for r, m in best)
        missing = [m for m in moduli if m not in selected_moduli]

        # Try multiple orderings of missing
        orderings = [
            missing[:],
            missing[::-1],
            sorted(missing, key=lambda m: len(find_valid_residues_fast(best, m))),
            sorted(missing, key=lambda m: -len(find_valid_residues_fast(best, m))),
        ]

        for order in orderings:
            if time.time() > time_limit:
                break
            current = list(best)
            current_moduli = set(m for r, m in current)
            for m in order:
                if time.time() > time_limit:
                    break
                if m in current_moduli:
                    continue
                valid = find_valid_residues_fast(current, m)
                if valid:
                    r = random.choice(valid)
                    current.append((r, m))
                    current_moduli.add(m)
            if len(current) > len(best):
                best = current
                selected_moduli = set(m for r, m in best)
                missing = [m for m in moduli if m not in selected_moduli]

        return best

    def local_search_remove_add(selected, time_limit, max_moves=3):
        """Try removing k elements and adding more to improve."""
        sol = [list(p) for p in selected]
        n = len(sol)

        # Try removing 1 element and adding 2+
        for i in range(n):
            if time.time() > time_limit:
                break
            temp = sol[:i] + sol[i+1:]
            temp_moduli = set(m for r, m in temp)

            # Try to add as many as possible
            added = 0
            for m in moduli:
                if time.time() > time_limit:
                    break
                if m in temp_moduli:
                    continue
                valid = find_valid_residues_fast(temp, m)
                if valid:
                    r = random.choice(valid)
                    temp.append((r, m))
                    temp_moduli.add(m)
                    added += 1

            if added > 1:
                sol = temp
                n = len(sol)

        # Try removing 2 elements and adding 3+
        if n >= 2:
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i+1, n):
                    if time.time() > time_limit:
                        break
                    temp = sol[:i] + sol[i+1:j] + sol[j+1:]
                    temp_moduli = set(m for r, m in temp)

                    added = 0
                    for m in moduli:
                        if time.time() > time_limit:
                            break
                        if m in temp_moduli:
                            continue
                        valid = find_valid_residues_fast(temp, m)
                        if valid:
                            r = random.choice(valid)
                            temp.append((r, m))
                            temp_moduli.add(m)
                            added += 1

                    if added > 2:
                        sol = temp
                        n = len(sol)
                        break

        return sol

    def swap_residue(selected, time_limit):
        """Try changing residues of existing elements to allow more additions."""
        sol = [list(p) for p in selected]
        n = len(sol)
        selected_moduli = set(m for r, m in sol)
        missing = [m for m in moduli if m not in selected_moduli]

        if not missing:
            return sol

        # Try changing one residue to see if it allows adding more
        for i in range(n):
            if time.time() > time_limit:
                break
            r_i, m_i = sol[i]
            # Try different residues for m_i
            for r_new in range(m_i):
                if r_new == r_i:
                    continue
                # Check if new residue is valid with others
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    r_j, m_j = sol[j]
                    g = gcd_table[m_i][m_j]
                    if (r_new - r_j) % g == 0:
                        ok = False
                        break
                if not ok:
                    continue

                # Try adding missing elements
                temp = sol[:]
                temp[i] = [r_new, m_i]
                temp_moduli = set(m for r, m in temp)
                added = 0
                for m in missing:
                    if time.time() > time_limit:
                        break
                    if m in temp_moduli:
                        continue
                    valid = find_valid_residues_fast(temp, m)
                    if valid:
                        r = random.choice(valid)
                        temp.append((r, m))
                        temp_moduli.add(m)
                        added += 1

                if added > 0:
                    sol = temp
                    n = len(sol)
                    selected_moduli = set(m for r, m in sol)
                    missing = [m for m in moduli if m not in selected_moduli]
                    break

        return sol

    # Initial solutions with different orderings
    best_solution = []
    best_size = 0

    # Precompute constraint tightness for ordering
    # A modulus with fewer valid residues (given no selections) is "tighter"
    # But initially all are valid, so use number of divisors as proxy
    def num_divisors(n):
        count = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i * i != n:
                    count += 1
            i += 1
        return count

    div_count = {m: num_divisors(m) for m in moduli}

    orderings = [
        moduli[:],  # ascending
        moduli[::-1],  # descending
        sorted(moduli, key=lambda m: div_count[m], reverse=True),  # most divisors first
        sorted(moduli, key=lambda m: div_count[m]),  # fewest divisors first
        sorted(moduli, key=lambda m: m),  # same as ascending
    ]

    for order in orderings:
        if time.time() > start_time + max_time - 5:
            break
        sol = greedy_with_order(order, randomize=False)
        sol = try_add_missing(sol, start_time + max_time - 3)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [list(p) for p in sol]

    # Randomized restarts
    while time.time() < start_time + max_time - 5:
        random.shuffle(moduli)
        current = greedy_with_order(moduli, randomize=True)
        current = try_add_missing(current, start_time + max_time - 3)

        # Local search
        if time.time() < start_time + max_time - 5:
            current = local_search_remove_add(current, start_time + max_time - 3)
            current = try_add_missing(current, start_time + max_time - 3)

        if len(current) > best_size:
            best_size = len(current)
            best_solution = [list(p) for p in current]

        # Also improve best solution
        if time.time() < start_time + max_time - 5:
            improved = local_search_remove_add(best_solution, start_time + max_time - 3)
            improved = try_add_missing(improved, start_time + max_time - 3)
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = [list(p) for p in improved]

    # Final polish: try to add any remaining missing
    if time.time() < start_time + max_time - 1:
        best_solution = try_add_missing(best_solution, start_time + max_time - 1)
        best_solution = swap_residue(best_solution, start_time + max_time - 1)
        best_solution = try_add_missing(best_solution, start_time + max_time - 1)

    if not best_solution:
        best_solution = [[0, 2]]

    return [[r, m] for r, m in best_solution]
