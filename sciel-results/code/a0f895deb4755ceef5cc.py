import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd table
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_tab[i][j] = math.gcd(moduli[i], moduli[j])

    # Precompute prime factor count for ordering
    def prime_factor_count(m):
        count = 0
        d = 2
        while d * d <= m:
            while m % d == 0:
                count += 1
                m //= d
            d += 1
        if m > 1:
            count += 1
        return count

    # Sort moduli by descending prime factor count (more factors = more conflicts earlier)
    order = sorted(range(n), key=lambda i: (-prime_factor_count(moduli[i]), moduli[i]))

    def get_valid_residues(selected, idx):
        """Get all valid residues for idx given current selection."""
        m = moduli[idx]
        # Start with all residues
        valid = [True] * m
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            if g == 1:
                continue
            base = sel_r % g
            for r in range(base, m, g):
                valid[r] = False
        return [r for r in range(m) if valid[r]]

    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if g > 1 and (new_r - r) % g == 0:
                return False
        return True

    def solve_backtrack(selected, remaining_indices, depth=0):
        """Backtracking with forward checking and MRV heuristic."""
        if time.time() > deadline - 3:
            return selected

        if not remaining_indices:
            return selected

        # MRV: choose variable with fewest valid residues
        best_idx = None
        best_residues = None
        min_count = float('inf')

        for idx in remaining_indices:
            residues = get_valid_residues(selected, idx)
            if not residues:
                return selected  # Dead end
            if len(residues) < min_count:
                min_count = len(residues)
                best_idx = idx
                best_residues = residues

        if best_idx is None:
            return selected

        # Try residues - prefer those that are "more constrained"
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            new_remaining = [idx for idx in remaining_indices if idx != best_idx]
            result = solve_backtrack(new_selected, new_remaining, depth + 1)
            if len(result) > len(selected):
                return result

        return selected

    def greedy_randomized(order_perm):
        """Greedy assignment with random ordering."""
        selected = []
        for idx in order_perm:
            m = moduli[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    break
        return selected

    def local_search(solution, max_iter=1000):
        """Local search: try to add missing moduli or swap to improve."""
        solution = list(solution)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            selected_idx_set = set(idx for (idx, _) in solution)

            # Try to add any missing modulus
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                valid_residues = get_valid_residues(solution, idx)
                if valid_residues:
                    solution.append((idx, valid_residues[0]))
                    selected_idx_set.add(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one, add two
            for i in range(len(solution)):
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)
                added_count = 0
                new_solution = list(reduced)
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(new_solution, idx)
                    if valid_residues:
                        new_solution.append((idx, valid_residues[0]))
                        reduced_idx_set.add(idx)
                        added_count += 1
                        if added_count >= 2:
                            break
                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    # Phase 1: Greedy random restarts
    random.seed(42)
    best_solution = []
    best_len = 0

    for restart in range(300):
        if time.time() > deadline - 25:
            break
        perm = list(range(n))
        random.shuffle(perm)
        sol = greedy_randomized(perm)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: Local search improvement
    if best_solution:
        improved_sol = local_search(best_solution, max_iter=2000)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Phase 3: Backtracking from best solution
    if time.time() < deadline - 10:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        remaining = [idx for idx in range(n) if idx not in selected_idx_set]
        # Use the prime-factor-count ordering for backtracking
        remaining_sorted = sorted(remaining, key=lambda i: (-prime_factor_count(moduli[i]), moduli[i]))
        bt_result = solve_backtrack(list(best_solution), remaining_sorted)
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)

    # Phase 4: More local search passes
    if time.time() < deadline - 5:
        improved_sol = local_search(best_solution, max_iter=5000)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Convert to required format [[r, m], ...]
    result = []
    for (idx, r) in best_solution:
        result.append([r, moduli[idx]])

    return result
