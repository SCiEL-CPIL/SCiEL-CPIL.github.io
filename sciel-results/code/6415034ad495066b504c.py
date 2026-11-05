import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit

    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)

    # Precompute gcds
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(candidates[i], candidates[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute number of distinct prime factors for ordering
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    prime_factor_counts = [num_distinct_prime_factors(m) for m in candidates]

    def get_forbidden_residues(selected, idx):
        """Get the set of forbidden residues for modulus at index idx."""
        m = candidates[idx]
        forbidden = set()
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            if g == 1:
                continue
            base = sel_r % g
            for r in range(base, m, g):
                forbidden.add(r)
        return forbidden

    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[new_idx][sel_idx]
            if (new_r - sel_r) % g == 0:
                return False
        return True

    def solve_greedy(order, randomize_residues=False):
        """Greedy assignment with given order."""
        selected = []
        for idx in order:
            m = candidates[idx]
            forbidden = get_forbidden_residues(selected, idx)
            valid_residues = [r for r in range(m) if r not in forbidden]
            if not valid_residues:
                continue
            if randomize_residues:
                random.shuffle(valid_residues)
            selected.append((idx, valid_residues[0]))
        return selected

    def local_search_improve(solution, max_iterations=50):
        """Improve solution by trying to add more moduli and swapping."""
        if time.time() > deadline - 5:
            return solution
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5 and iterations < max_iterations:
            iterations += 1
            improved = False
            selected_idx_set = set(idx for (idx, _) in solution)
            
            # Try to add any missing moduli
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                forbidden = get_forbidden_residues(solution, idx)
                m = candidates[idx]
                valid_residues = [r for r in range(m) if r not in forbidden]
                if valid_residues:
                    solution.append((idx, valid_residues[0]))
                    selected_idx_set.add(idx)
                    improved = True
                    break
            
            if not improved and time.time() < deadline - 10:
                # Try swap: remove one, see if we can add two
                for i in range(len(solution)):
                    if time.time() > deadline - 10:
                        break
                    removed = solution[i]
                    reduced = solution[:i] + solution[i+1:]
                    reduced_idx_set = set(idx for (idx, _) in reduced)
                    
                    added_count = 0
                    new_solution = list(reduced)
                    for idx in range(n):
                        if idx in reduced_idx_set:
                            continue
                        forbidden = get_forbidden_residues(new_solution, idx)
                        m = candidates[idx]
                        valid_residues = [r for r in range(m) if r not in forbidden]
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

    def backtracking_search(selected, available_indices, time_limit):
        """Backtracking with forward checking and MRV heuristic."""
        if time.time() > time_limit:
            return selected
        
        if not available_indices:
            return selected
        
        # MRV: choose index with fewest valid residues
        best_idx = None
        best_residues = None
        for idx in available_indices:
            forbidden = get_forbidden_residues(selected, idx)
            m = candidates[idx]
            valid_residues = [r for r in range(m) if r not in forbidden]
            if not valid_residues:
                continue
            if best_residues is None or len(valid_residues) < len(best_residues):
                best_idx = idx
                best_residues = valid_residues
        
        if best_idx is None:
            return selected
        
        # Try each residue
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            remaining = [idx for idx in available_indices if idx != best_idx]
            result = backtracking_search(new_selected, remaining, time_limit)
            if len(result) > len(selected):
                return result
        
        return selected

    best_solution = []
    best_len = 0

    # Strategy 1: Greedy with different orderings
    # Order by descending number of prime factors (harder first for early conflict)
    order_desc_pf = sorted(range(n), key=lambda i: (-prime_factor_counts[i], candidates[i]))
    sol = solve_greedy(order_desc_pf)
    if len(sol) > best_len:
        best_solution = sol
        best_len = len(sol)

    # Order by ascending number of prime factors
    order_asc_pf = sorted(range(n), key=lambda i: (prime_factor_counts[i], candidates[i]))
    sol = solve_greedy(order_asc_pf)
    if len(sol) > best_len:
        best_solution = sol
        best_len = len(sol)

    # Random restarts
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        order = list(range(n))
        random.shuffle(order)
        sol = solve_greedy(order, randomize_residues=True)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
        if restart_count >= 200:
            break

    # Local search improvement
    if best_solution and time.time() < deadline - 10:
        improved_sol = local_search_improve(list(best_solution))
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Backtracking from best solution
    if time.time() < deadline - 5:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]
        if available:
            bt_result = backtracking_search(list(best_solution), available, deadline - 2)
            if len(bt_result) > best_len:
                best_solution = bt_result
                best_len = len(best_solution)

    # Convert to required format [[r, m], ...]
    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])

    # Ensure non-empty
    if not result:
        result = [[1, 2]]

    return result
