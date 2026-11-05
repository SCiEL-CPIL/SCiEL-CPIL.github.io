"""
Combined approach: greedy randomized + local search + backtracking
for maximum disjoint residue classes with distinct moduli 2..128.
"""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)
    
    # Precompute gcd table
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])
    
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
            base = sel_r % g
            for r in range(base, m, g):
                valid.discard(r)
        return sorted(valid)
    
    def greedy_randomized(order, seed):
        """Greedy assignment with random residue order."""
        random.seed(seed)
        selected = []
        for idx in order:
            m = candidates[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    break
        return selected
    
    def local_search_improve(solution, max_iters=500):
        """Improve solution by trying to add missing moduli or swap."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            selected_idx_set = set(idx for (idx, _) in solution)
            
            # Try to add any missing modulus
            added_any = False
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                valid_residues = get_valid_residues(solution, idx)
                if valid_residues:
                    solution.append((idx, valid_residues[0]))
                    selected_idx_set.add(idx)
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # Try swap: remove one, try to add two
            for i in range(len(solution)):
                removed = solution[i]
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)
                
                added = 0
                new_solution = list(reduced)
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(new_solution, idx)
                    if valid_residues:
                        new_solution.append((idx, valid_residues[0]))
                        reduced_idx_set.add(idx)
                        added += 1
                        if added >= 2:
                            break
                
                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break
        
        return solution
    
    # Phase 1: Greedy + random restarts to get good initial solutions
    best_solution = []
    best_len = 0
    
    # Try multiple ordering strategies
    strategies = []
    
    # Random orderings
    for seed in range(100):
        if time.time() > deadline - 40:
            break
        order = list(range(n))
        random.seed(seed)
        random.shuffle(order)
        strategies.append((order, seed))
    
    # Ascending order
    strategies.append((list(range(n)), 99999))
    
    # Descending order
    strategies.append((list(range(n, -1, -1)), 88888))
    
    # Multiples of 6
    mult6 = [i for i in range(n) if candidates[i] % 6 == 0]
    strategies.append((mult6, 12345))
    
    # Multiples of 2
    mult2 = [i for i in range(n) if candidates[i] % 2 == 0]
    strategies.append((mult2, 54321))
    
    # Sort by number of divisors (more divisors = more constraints, try first)
    def num_divisors(m):
        count = 0
        for i in range(2, int(math.sqrt(m)) + 1):
            if m % i == 0:
                count += 2
                if i * i == m:
                    count -= 1
        return count + 1
    
    by_divisors = sorted(range(n), key=lambda i: num_divisors(candidates[i]), reverse=True)
    strategies.append((by_divisors, 77777))
    
    # Sort by modulus size descending
    by_size_desc = sorted(range(n), key=lambda i: candidates[i], reverse=True)
    strategies.append((by_size_desc, 66666))
    
    for order, seed in strategies:
        if time.time() > deadline - 35:
            break
        sol = greedy_randomized(order, seed)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Phase 2: Local search improvement
    if best_solution:
        improved_sol = local_search_improve(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(improved_sol)
    
    # Phase 3: Backtracking to try to add more
    if time.time() < deadline - 20 and best_solution:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]
        
        def backtrack_add(selected, available_indices, depth=0):
            if time.time() > deadline - 10 or depth > 20:
                return selected
            if not available_indices:
                return selected
            
            # MRV: pick index with fewest valid residues
            best_idx = None
            best_residues = None
            min_count = float('inf')
            for idx in available_indices:
                residues = get_valid_residues(selected, idx)
                if not residues:
                    continue
                if len(residues) < min_count:
                    min_count = len(residues)
                    best_idx = idx
                    best_residues = residues
            
            if best_idx is None:
                return selected
            
            for r in best_residues:
                new_selected = selected + [(best_idx, r)]
                remaining = [idx for idx in available_indices if idx != best_idx]
                result = backtrack_add(new_selected, remaining, depth + 1)
                if len(result) > len(selected):
                    return result
            
            return selected
        
        bt_result = backtrack_add(list(best_solution), available)
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)
    
    # Final validation: ensure all pairs are truly disjoint
    def validate(solution):
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                idx1, r1 = solution[i]
                idx2, r2 = solution[j]
                g = gcd_tab[idx1][idx2]
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    if not validate(best_solution):
        # Repair: remove conflicting pairs
        repaired = []
        for item in best_solution:
            if all(is_compatible(repaired, item[0], item[1]) for _ in [1]):
                repaired.append(item)
        best_solution = repaired
    
    # Convert to required format [[r, m], ...]
    result = [[r, candidates[idx]] for (idx, r) in best_solution]
    return result
