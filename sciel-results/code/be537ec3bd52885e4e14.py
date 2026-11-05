import time
import math
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

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

    # Precompute conflict graph: edge between i,j if gcd(i,j) > 1
    # (they might conflict depending on residues)
    conflict_edges = {}
    for i in range(n):
        conflict_edges[i] = []
    for i in range(n):
        for j in range(i+1, n):
            if gcd_tab[i][j] > 1:
                conflict_edges[i].append(j)
                conflict_edges[j].append(i)

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
            # Try residues in order of least conflict first
            # For each residue, count how many selected it conflicts with
            best_r = None
            min_conflicts = m + 1
            for r in range(m):
                conflicts = 0
                for (idx2, r2, m2) in selected:
                    g = gcd_tab[idx][idx2]
                    if (r - r2) % g == 0:
                        conflicts += 1
                if conflicts == 0:
                    best_r = r
                    break
                elif conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_r = r
            if best_r is not None:
                selected.append((idx, best_r, m))
        return selected

    def backtracking_search(order, limit):
        """Backtracking search with pruning. Returns best solution found."""
        best = []
        
        def dfs(pos, current):
            nonlocal best
            if len(current) > len(best):
                best = list(current)
            if pos == len(order) or len(current) >= limit:
                return
            idx = order[pos]
            m = moduli[idx]
            for r in range(m):
                if check_residue(idx, r, current):
                    current.append((idx, r, m))
                    dfs(pos + 1, current)
                    current.pop()
        
        dfs(0, [])
        return best

    best = []
    best_len = 0

    # Order 1: by number of conflicts (descending) - most constrained first
    order_by_conflicts = sorted(range(n), key=lambda i: -len(conflict_edges[i]))
    
    # Order 2: by modulus (ascending)
    order_asc = sorted(range(n), key=lambda i: moduli[i])
    
    # Order 3: by modulus (descending)
    order_desc = sorted(range(n), key=lambda i: -moduli[i])
    
    # Order 4: prefer odd moduli, then by conflict count
    order_odd_first = sorted(range(n), key=lambda i: (moduli[i] % 2, -len(conflict_edges[i])))
    
    # Order 5: prefer prime moduli
    def is_prime(m):
        if m < 2:
            return False
        for d in range(2, int(math.sqrt(m)) + 1):
            if m % d == 0:
                return False
        return True
    order_prime_first = sorted(range(n), key=lambda i: (not is_prime(moduli[i]), -len(conflict_edges[i])))
    
    # Order 6: by number of distinct prime factors (ascending)
    def num_distinct_prime_factors(m):
        count = 0
        d = 2
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    order_few_factors = sorted(range(n), key=lambda i: (num_distinct_prime_factors(moduli[i]), -len(conflict_edges[i])))
    
    orderings = [order_by_conflicts, order_asc, order_desc, order_odd_first, order_prime_first, order_few_factors]
    
    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts
    random.seed(42)
    for _ in range(3000):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Local search: try to add more moduli
    if time.time() < deadline - 20:
        selected_idx = set(idx for (idx, _, _) in best)
        improved = True
        while improved and time.time() < deadline - 20:
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
    if time.time() < deadline - 10:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 10:
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

    # Try to improve by removing one and adding two
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        for remove_pos in range(len(best)):
            if time.time() > deadline - 5:
                break
            remove_idx, remove_r, remove_m = best[remove_pos]
            remaining = [b for i, b in enumerate(best) if i != remove_pos]
            remaining_idx = set(idx for (idx, _, _) in remaining)
            
            added = 0
            for new_idx in range(n):
                if time.time() > deadline - 5:
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
            
            if added > 1:
                best = remaining
                best_len = len(best)
                selected_idx = set(idx for (idx, _, _) in best)

    return [[r, m] for (_, r, m) in best]
