import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    # Consider ALL moduli from 2 to 128, not just even ones
    candidates = list(range(2, MAX_M + 1))
    
    n = len(candidates)
    
    # Precompute gcds for the candidate set
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])
    
    # Precompute for each modulus m, the number of residues r in [0,m) 
    # that are compatible with a given (r2, m2)
    # For a pair (m1, m2) with g = gcd(m1,m2), exactly g residues r1 in [0,m1)
    # satisfy (r1 - r2) % g == 0, so m1 - g residues are valid.
    
    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if (new_r - r) % g == 0:
                return False
        return True
    
    def get_valid_residues(selected, idx):
        """Get all valid residues for idx given current selection."""
        m = candidates[idx]
        valid = set(range(m))
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            # Remove residues r where (r - sel_r) % g == 0
            # These are: sel_r % g, (sel_r % g) + g, (sel_r % g) + 2g, ... < m
            base = sel_r % g
            for r in range(base, m, g):
                if r in valid:
                    valid.discard(r)
        return sorted(valid)
    
    def solve_backtrack(selected, available_indices, depth=0):
        """Backtracking search with constraint propagation."""
        if time.time() > deadline - 5:
            return selected
        
        if not available_indices:
            return selected
        
        # Choose the index with fewest valid residues (MRV heuristic)
        best_idx = None
        best_residues = None
        for idx in available_indices:
            residues = get_valid_residues(selected, idx)
            if not residues:
                continue
            if best_residues is None or len(residues) < len(best_residues):
                best_idx = idx
                best_residues = residues
        
        if best_idx is None:
            return selected
        
        # Try residues in order
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            remaining = [idx for idx in available_indices if idx != best_idx]
            result = solve_backtrack(new_selected, remaining, depth + 1)
            if len(result) > len(selected):
                return result
        
        return selected
    
    def solve_greedy_randomized(order):
        selected = []
        for idx in order:
            m = candidates[idx]
            found = False
            # Try residues in random order to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    found = True
                    break
        return selected
    
    def local_search_improve(solution):
        """Try to improve solution by swapping or adding."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            selected_idx_set = set(idx for (idx, _) in solution)
            
            # Try to add any missing moduli
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
            
            # Try to swap: remove one and add two
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    # Try removing solution[i] and solution[j], then see if we can add more
                    removed = [solution[i], solution[j]]
                    reduced = [x for k, x in enumerate(solution) if k != i and k != j]
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
                if improved:
                    break
        
        return solution
    
    best_solution = []
    best_len = 0
    
    # First: greedy random restarts to get a good initial solution
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = solve_greedy_randomized(order)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Local search improvement
    if best_solution:
        improved_sol = local_search_improve(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)
    
    # Backtracking search from the best solution found
    if time.time() < deadline - 10:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]
        
        # Try backtracking to see if we can do better
        bt_result = solve_backtrack(list(best_solution), available)
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)
    
    # Convert to required format [[r, m], ...]
    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])
        
    return result
