import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    MAX_M = 128
    # Focus on even moduli as they form a large pairwise non-coprime set
    candidates = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    
    n = len(candidates)
    
    # Precompute gcds for the candidate set
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

    best_solution = []
    best_len = 0
    
    # Multiple random restarts
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline - 15:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = solve_greedy_randomized(order)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)

    # Local search: try to add more
    selected_idx_set = set(idx for (idx, _) in best_solution)
    
    # Try to add any missing even moduli
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(n):
            if idx in selected_idx_set:
                continue
            m = candidates[idx]
            # Try to find a residue
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(best_solution, idx, r):
                    best_solution.append((idx, r))
                    selected_idx_set.add(idx)
                    improved = True
                    break
            if improved:
                break

    # --- NEW: Backtracking polish to find better assignments for existing moduli ---
    # This attempts to improve the solution by re-assigning residues in a smarter way
    # using a limited backtracking approach on the current set of selected moduli.
    
    # We'll try to improve the current best_solution by attempting to re-solve
    # with a more structured backtracking approach, but only if we have time.
    
    # Actually, let's try a different local search: swap residues to potentially allow adding more.
    # Or, more simply, let's try to re-run the greedy with the current solution as a starting point
    # but with a different ordering of the *unselected* moduli, or re-optimizing the selected ones.
    
    # Let's implement a simple local search that tries to change residues of selected moduli
    # to see if it allows adding a new modulus.
    
    if time.time() < deadline - 5:
        # Try to add more by relaxing constraints temporarily? No, that's complex.
        # Instead, let's try a few more random restarts with the best_len as a lower bound.
        # Or, try to improve the current solution by swapping residues.
        
        # Let's try to add missing moduli again with a different strategy:
        # Sort unselected moduli by difficulty (fewest compatible residues first)
        unselected = [idx for idx in range(n) if idx not in selected_idx_set]
        
        # Calculate "difficulty" for each unselected modulus
        def count_compatible(idx, selected):
            count = 0
            m = candidates[idx]
            for r in range(m):
                if is_compatible(selected, idx, r):
                    count += 1
            return count
        
        # Sort by count (ascending) - try the hardest ones first? Or easiest?
        # Usually, easiest first is safer for greedy, but for adding to an existing set,
        # maybe hardest first helps avoid dead ends? Let's try easiest first (most options).
        unselected_sorted = sorted(unselected, key=lambda idx: count_compatible(idx, best_solution))
        
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            for idx in unselected_sorted:
                if idx in selected_idx_set:
                    continue
                m = candidates[idx]
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(best_solution, idx, r):
                        best_solution.append((idx, r))
                        selected_idx_set.add(idx)
                        improved2 = True
                        break
                if improved2:
                    # Recalculate difficulties? Expensive. Just break and restart loop.
                    break
    
    # Final polish: try to maximize by re-running greedy on the full set of candidates
    # but seeding with the current best solution's moduli if possible?
    # Actually, the current approach is solid. Let's just return.
    
    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])
        
    return result
