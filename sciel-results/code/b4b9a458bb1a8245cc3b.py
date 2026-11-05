import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute prime counts for sorting
    def prime_factor_count(m):
        count = 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if m % p == 0:
                count += 1
        return count

    pf_counts = [prime_factor_count(moduli[i]) for i in range(n)]

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2, m2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_greedy(order, residue_order_func=None):
        selected = []
        for idx in order:
            m = moduli[idx]
            if residue_order_func is None:
                # Try residue 1 first, then others
                residues = [1] + [r for r in range(m) if r != 1]
            else:
                residues = residue_order_func(m)
            found = False
            for r in residues:
                if check_residue(idx, r, selected):
                    selected.append((idx, r, m))
                    found = True
                    break
        return selected

    best = []
    best_len = 0

    # Multiple deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: -pf_counts[i]),  # harder first
        sorted(range(n), key=lambda i: pf_counts[i]),   # easier first
        # Prefer moduli with fewer small prime factors (easier to place)
        sorted(range(n), key=lambda i: (pf_counts[i], moduli[i])),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts with both fixed and random residue orders
    random.seed(12345)
    
    def random_residue_order(m):
        res = list(range(m))
        random.shuffle(res)
        return res

    # First pass: random orderings with fixed residue order
    for _ in range(3000):
        if time.time() > deadline - 25:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Second pass: random orderings with random residue order
    for _ in range(2000):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order, residue_order_func=random_residue_order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Local search: try to add more moduli
    if time.time() < deadline - 15:
        selected_idx = set(idx for (idx, _, _) in best)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue(idx, r, best):
                        best.append((idx, r, m))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    # Local search: try swapping residues to enable adding new moduli
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                # Try changing one existing residue to free up space for new_idx
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    # Try all residues for old_idx
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        # Check if new_r is compatible with all other selected (excluding old_idx)
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r, other_m = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        # Now check if new_idx can be added with some residue
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r, other_m = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            best[pos] = (old_idx, new_r, old_m)
                            best.append((new_idx, r_new, m_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

    return [[r, m] for (_, r, m) in best]
