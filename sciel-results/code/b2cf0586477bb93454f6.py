import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (5s buffer)
    
    # Candidate moduli: All integers in [20, 200]
    # We want to maximize sum(1/m), so smaller moduli are more valuable.
    # However, smaller moduli are more constrained (fewer residues, more conflicts).
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcds for speed
    # gcd_cache[m1][m2]
    # Since moduli are up to 200, a 201x201 matrix is small.
    gcd_cache = [[0]*201 for _ in range(201)]
    for m1 in all_moduli:
        for m2 in all_moduli:
            gcd_cache[m1][m2] = gcd(m1, m2)
            
    # State:
    # selected: list of (m, r)
    # We want to select a subset of moduli and assign residues.
    
    # Strategy:
    # 1. Start with an empty set.
    # 2. Use a backtracking solver to find a valid assignment for a subset of moduli.
    #    To maximize score, we should prefer smaller moduli.
    #    However, simply adding all moduli is impossible.
    #    We can frame this as: Try to include moduli in order of increasing m (highest weight).
    #    If we can't fit a modulus, we might skip it or try to rearrange existing ones.
    
    # Better Strategy: 
    # The problem is equivalent to finding a maximum weight independent set in a graph where 
    # nodes are (m, r) and edges connect conflicting pairs? No, that's huge.
    # 
    # Let's use a heuristic search:
    # 1. Initialize with a greedy solution (similar to current but smarter).
    # 2. Perform local search: 
    #    - Try to add a missing modulus by changing residues of existing ones.
    #    - Try to swap a low-value modulus for a higher-value one.
    
    def is_valid_addition(m, r, selected):
        for m_sel, r_sel in selected:
            g = gcd_cache[m][m_sel]
            if r % g == r_sel % g:
                return False
        return True
        
    def get_conflicts(m, r, selected):
        conflicts = []
        for i, (m_sel, r_sel) in enumerate(selected):
            g = gcd_cache[m][m_sel]
            if r % g == r_sel % g:
                conflicts.append(i)
        return conflicts

    # Phase 1: Greedy Construction with MRV-like heuristic
    # We process moduli from smallest to largest (highest weight first).
    # For each modulus, we try to find a residue.
    # To improve on simple greedy, we can look ahead: if a residue choice for m 
    # blocks many future small moduli, it's bad. But calculating this is expensive.
    
    # Let's try a different greedy: 
    # Sort moduli by 1/m (descending). 
    # For each m, try all residues r. 
    # Score each r by how many future moduli (larger m) it conflicts with? 
    # Or simpler: Just try to fit as many as possible.
    
    # Let's implement a backtracking solver that tries to include moduli in order of m.
    # If it fails to include m, it skips it.
    # This is effectively a greedy with backtracking for the current step.
    
    def solve_greedy_backtrack(moduli_list):
        selected = []
        # moduli_list should be sorted by m
        for m in moduli_list:
            # Try all residues for m
            found = False
            # Order residues: maybe 0 is best? Or random?
            # Let's try 0 first, then others.
            for r in range(m):
                if is_valid_addition(m, r, selected):
                    selected.append((m, r))
                    found = True
                    break
            # If no residue works, we skip this modulus.
            # (We could backtrack and change previous residues, but that's expensive.
            #  However, the local search phase will handle this.)
        return selected

    # Initial solution
    current_solution = solve_greedy_backtrack(all_moduli)
    
    # Calculate score
    def calc_score(sol):
        return sum(1.0/m for m, r in sol)
        
    current_score = calc_score(current_solution)
    best_solution = current_solution
    best_score = current_score
    
    # Phase 2: Local Search / Repair
    # We have time. Let's try to improve.
    # Moves:
    # 1. Add a modulus: Pick an m not in solution. Try to find a residue r.
    #    If no r works, try to change residues of existing moduli to make space.
    # 2. Swap: Remove a large m, try to add a smaller m.
    
    # To make "Add" work, we need to be able to change existing residues.
    # This is a CSP re-solve on the existing set + new m.
    
    def try_add_modulus(m_new, selected):
        # Returns a new solution if m_new can be added, else None
        # We use backtracking to adjust existing residues if needed.
        # This is expensive, so we limit depth or use heuristics.
        
        # Heuristic: Try to find a residue for m_new that conflicts with fewest existing.
        # Then, for those conflicts, try to reassign their residues.
        
        # Simpler approach: Random restarts of the assignment for the whole set + new m?
        # No, let's do a limited backtracking on the conflicting subset.
        
        # For now, a simple repair:
        # If m_new can be added with current residues, do it.
        for r_new in range(m_new):
            if is_valid_addition(m_new, r_new, selected):
                new_sol = selected + [(m_new, r_new)]
                return new_sol
                
        # If not, try to modify one existing modulus to free up space.
        # Pick a modulus m_old in selected that conflicts with all possible r_new?
        # Or just pick a random modulus and try to reassign it.
        
        # This is getting complex. Let's use a simpler local search:
        # Randomly perturb residues of existing moduli and see if we can add a new one.
        
        return None

    # A more effective local search for this specific problem:
    # The "disjointness" constraint is local to pairs.
    # We can use a "simulated annealing" like approach or just random restarts with greedy.
    
    # Given the 115s budget, we can afford many iterations.
    
    # Strategy: 
    # 1. Start with the greedy solution.
    # 2. Loop:
    #    - Pick a random modulus m_add from those not in solution (prioritize small m).
    #    - Try to insert it by re-solving the CSP for (current_set + m_add) using a fast heuristic.
    #    - If successful and score improves, accept.
    #    - Also, try to remove a large modulus and replace with a smaller one.
    
    # To make insertion feasible, we need a fast CSP solver for a fixed set of moduli.
    # Let's implement a simple backtracking solver for a given list of moduli.
    
    def solve_csp(moduli_set):
        # Returns a valid assignment for moduli_set or None
        # Order moduli by m (smallest first) for better pruning?
        # Actually, MRV (minimum remaining values) is better.
        # But calculating MRV dynamically is costly.
        # Let's just sort by m.
        
        moduli_sorted = sorted(moduli_set)
        selected = []
        
        def backtrack(idx):
            if idx == len(moduli_sorted):
                return True
            m = moduli_sorted[idx]
            # Try residues
            # Heuristic: Try residues that conflict with fewest previous?
            # For now, just try 0..m-1
            for r in range(m):
                if is_valid_addition(m, r, selected):
                    selected.append((m, r))
                    if backtrack(idx + 1):
                        return True
                    selected.pop()
            return False
            
        if backtrack(0):
            return selected
        return None

    # Use this solver to improve the solution.
    # We want to find a subset of moduli that maximizes sum(1/m).
    # This is like a knapsack problem with complex constraints.
    
    # Heuristic: 
    # Iterate through potential additions.
    # Keep a set of "promising" moduli (small ones).
    
    # Let's try a different angle:
    # The greedy solution includes many moduli. 
    # We can try to "swap" a large modulus m_large with a smaller m_small that was skipped.
    
    # 1. Identify skipped small moduli.
    selected_moduli_set = {m for m, r in best_solution}
    skipped_moduli = [m for m in all_moduli if m not in selected_moduli_set]
    # Sort skipped by 1/m (descending) i.e., m ascending
    skipped_moduli.sort()
    
    # 2. Identify large moduli in solution (candidates for removal)
    selected_list = list(best_solution)
    selected_list.sort(key=lambda x: x[0], reverse=True) # Largest first
    
    # 3. Attempt swaps
    # For each small skipped m, try to add it by removing one large selected m.
    # We need to check if the remaining set + m_small is solvable.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a skipped modulus by removing a selected one
        # We prioritize adding small m and removing large m.
        
        for m_add in skipped_moduli[:10]: # Only consider top 10 smallest skipped
            if m_add in selected_moduli_set:
                continue
                
            # Try to remove each selected modulus (from largest to smallest)
            # and see if the rest + m_add is solvable.
            
            # To speed up, we can check if m_add is "compatible" with the core.
            # Let's just try the swap.
            
            # Create a list of moduli to solve: (current_set - {m_remove}) + {m_add}
            
            # We iterate m_remove from largest to smallest
            for m_remove in [m for m, r in selected_list if m > m_add]:
                # Candidate set
                cand_moduli = [m for m, r in best_solution if m != m_remove]
                cand_moduli.append(m_add)
                
                # Solve CSP for cand_moduli
                # This might be slow if done naively.
                # Optimization: Only try this if m_add is significantly smaller than m_remove.
                if 1.0/m_add - 1.0/m_remove < 0.001:
                    continue # Not worth the effort if gain is tiny
                
                sol = solve_csp(cand_moduli)
                if sol is not None:
                    new_score = calc_score(sol)
                    if new_score > best_score:
                        best_solution = sol
                        best_score = new_score
                        selected_moduli_set = {m for m, r in best_solution}
                        # Update skipped and selected lists
                        selected_list = list(best_solution)
                        selected_list.sort(key=lambda x: x[0], reverse=True)
                        skipped_moduli = [m for m in all_moduli if m not in selected_moduli_set]
                        skipped_moduli.sort()
                        improved = True
                        break # Restart the m_remove loop with new state
            if improved:
                break
                
        # Also try simple additions without removal (if we skipped some due to greedy failure)
        # This is handled by the swap logic if we consider removing nothing? 
        # No, swap requires removal. 
        # Let's try direct addition of skipped moduli if possible without removal.
        # This is what the greedy did, but maybe residues can be adjusted.
        
        # Try to add m_add to current solution by re-solving CSP for (current + m_add)
        # This is expensive. Let's limit this.
        
        for m_add in skipped_moduli[:5]:
            if m_add in selected_moduli_set:
                continue
            cand_moduli = [m for m, r in best_solution]
            cand_moduli.append(m_add)
            sol = solve_csp(cand_moduli)
            if sol is not None:
                new_score = calc_score(sol)
                if new_score > best_score:
                    best_solution = sol
                    best_score = new_score
                    selected_moduli_set = {m for m, r in best_solution}
                    selected_list = list(best_solution)
                    selected_list.sort(key=lambda x: x[0], reverse=True)
                    skipped_moduli = [m for m in all_moduli if m not in selected_moduli_set]
                    skipped_moduli.sort()
                    improved = True
                    break
        if improved:
            break
            
    # Final check: Ensure we return a valid format
    result = [[m, r] for m, r in best_solution]
    
    if not result:
        result = [[20, 0]]
        
    return result

# For testing/debugging if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Selected {len(res)} moduli")
    print(f"Score: {sum(1.0/m for m, r in res)}")
    # Verify validity
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                print(f"Invalid: {res[i]} and {res[j]}")
