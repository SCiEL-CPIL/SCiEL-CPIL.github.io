import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

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

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2, m2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_greedy(order):
        selected = []
        for idx in order:
            m = moduli[idx]
            found = False
            # Try residue 1 first, then others in a pseudo-random but deterministic order
            residues = [1] + [r for r in range(m) if r != 1]
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
        sorted(range(n), key=lambda i: math.gcd(moduli[i], 2) * 10 + moduli[i] % 2),
        sorted(range(n), key=lambda i: -sum(1 for p in [2,3,5,7,11,13,17,19,23] if moduli[i] % p == 0)),
        # New: prefer moduli with fewer divisors (prime-like) first
        sorted(range(n), key=lambda i: sum(1 for d in range(2, moduli[i]+1) if moduli[i] % d == 0)),
        # New: prefer odd moduli first
        sorted(range(n), key=lambda i: moduli[i] % 2),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts with more iterations
    random.seed(12345)
    for _ in range(5000):
        if time.time() > deadline - 25:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
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
    if time.time() < deadline - 8:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 8:
            swap_improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
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

    # Final polish: try more aggressive local search with multiple swap attempts
    if time.time() < deadline - 3:
        selected_idx = set(idx for (idx, _, _) in best)
        # Try to remove one and add two
        for remove_pos in range(len(best)):
            if time.time() > deadline - 3:
                break
            if not selected_idx:
                break
            remove_idx, remove_r, remove_m = best[remove_pos]
            remaining = [b for i, b in enumerate(best) if i != remove_pos]
            remaining_idx = set(idx for (idx, _, _) in remaining)
            
            # Try to add two new moduli
            added = 0
            for new_idx in range(n):
                if time.time() > deadline - 3:
                    break
                if new_idx in remaining_idx or new_idx == remove_idx:
                    continue
                m_new = moduli[new_idx]
                for r_new in range(m_new):
                    if check_residue(new_idx, r_new, remaining):
                        remaining.append((new_idx, r_new, m_new))
                        remaining_idx.add(new_idx)
                        added += 1
                        break
                if added >= 2:
                    break
            
            if added >= 1:
                best = remaining
                best_len = len(best)
                selected_idx = set(idx for (idx, _, _) in best)

    return [[r, m] for (_, r, m) in best]
