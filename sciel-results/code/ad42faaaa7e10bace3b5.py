"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # Pre-compute GCDs
    gcd_matrix = [[0] * num_moduli for _ in range(num_moduli)]
    for i in range(num_moduli):
        m_i = all_moduli[i]
        for j in range(i, num_moduli):
            m_j = all_moduli[j]
            g = math.gcd(m_i, m_j)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Helper: Score calculation
    def calc_score(indices):
        return sum(1.0 / all_moduli[i] for i in indices)

    # Helper: Check if adding (m_idx, r) to a set of (m_idx, r) is valid
    def is_addition_valid(new_m_idx, new_r, existing_list):
        for m2_idx, r2 in existing_list:
            g = gcd_matrix[new_m_idx][m2_idx]
            if (new_r - r2) % g == 0:
                return False
        return True

    # Helper: Find a valid residue for m_idx given a set of (m_idx, r)
    def find_valid_residue(m_idx, existing_list):
        m = all_moduli[m_idx]
        # Try residues in random order? Or 0..m-1?
        # Random order might find "sparse" solutions faster in some contexts,
        # but deterministic 0..m-1 is often faster if a solution exists.
        # Let's try random first for diversity, then fallback to sequential if needed?
        # Actually, for CSP solving, we want ANY valid residue.
        # Let's try 0..m-1. If none found, no solution for this assignment order.
        # Wait, we need backtracking.
        # For now, this is a "forward checking" helper for greedy/SA.
        for r in range(m):
            if is_addition_valid(m_idx, r, existing_list):
                return r
        return -1

    # Helper: Full CSP solver with backtracking for a given set of modulus indices
    def solve_csp(indices, time_limit=1.0):
        # indices: list of indices into all_moduli
        # Returns: list of (m_idx, r) or None
        # We want to assign residues to all indices such that they are pairwise disjoint.
        
        # Order indices: Most constrained first? 
        # High GCD with others is more constraining.
        # Or just small m first (smaller domain? No, domain size is m).
        # Actually, small m has smaller domain [0, m-1], so more likely to fail?
        # Let's sort by m ascending (smaller domain).
        sorted_indices = sorted(indices, key=lambda i: all_moduli[i])
        
        n = len(sorted_indices)
        assignment = [None] * n
        
        def backtrack(k):
            if time.time() > start_time + time_limit:
                return False
            if k == n:
                return True
            
            m_idx = sorted_indices[k]
            m = all_moduli[m_idx]
            
            # Try residues 0..m-1
            # Optimization: If we have many constraints, we can filter residues early.
            # But simple loop is okay for m <= 200.
            
            # To make it more robust, shuffle residues?
            # Deterministic is faster if solution is at 0.
            # But if we get stuck, we might want random.
            # For now, try 0..m-1.
            
            for r in range(m):
                # Check against already assigned
                valid = True
                for j in range(k):
                    m2_idx = sorted_indices[j]
                    r2 = assignment[j]
                    g = gcd_matrix[m_idx][m2_idx]
                    if (r - r2) % g == 0:
                        valid = False
                        break
                
                if valid:
                    assignment[k] = r
                    if backtrack(k + 1):
                        return True
                    assignment[k] = None
            
            return False

        if backtrack(0):
            return [(sorted_indices[i], assignment[i]) for i in range(n)]
        return None

    # Strategy 1: Greedy construction
    def greedy_construct():
        state = []
        # Sort by 1/m descending (i.e., m ascending)
        # We want to add the most valuable ones first.
        for m_idx in range(num_moduli):
            m = all_moduli[m_idx]
            r = find_valid_residue(m_idx, state)
            if r != -1:
                state.append((m_idx, r))
            else:
                # Could try harder here, but greedy is fast
                pass
        return state

    # Strategy 2: SA on the SET of moduli, solving CSP for each set
    # This is expensive. 
    # Alternative: SA on the RESIDUES for a FIXED set? 
    # Or SA on both?
    
    # Let's stick to a hybrid:
    # 1. Start with greedy.
    # 2. Local Search: 
    #    - Try to add a missing modulus (find valid residue).
    #    - Try to swap a modulus (remove one, add another).
    #    - Try to change a residue (if it allows adding another? No, score doesn't change for residue change unless it enables addition).
    #    Actually, residue changes don't change score, only feasibility.
    #    So the score is purely determined by the SET of moduli.
    #    Therefore, we should optimize the SET of moduli.
    
    # New Strategy: 
    # Maximize sum(1/m) over valid sets S.
    # A set S is valid if there exists an assignment of residues.
    # Checking validity is NP-hard (CSP), but for small sets it's fast.
    
    # Let's use a Greedy Set Construction with Backtracking/Repair.
    
    best_state = greedy_construct()
    best_score = calc_score([m_idx for m_idx, r in best_state])
    
    # Local Search on the Set
    # Moves:
    # 1. Add a random unselected modulus. Check if CSP is solvable.
    # 2. Remove a random selected modulus. (Always valid, decreases score).
    # 3. Swap: Remove one, Add one.
    
    # To speed up, we don't want to run full CSP for every move.
    # We can maintain the current assignment.
    
    current_state = best_state[:]
    current_score = best_score
    
    # We need a way to efficiently check if we can add a new modulus to the current assignment.
    # If we change residues, we need to re-solve or adjust.
    
    # Let's try a different angle: 
    # Since the score depends only on the set, and valid sets are dense,
    # maybe we can just pick a large set of moduli that are likely to be mutually disjoint?
    # Moduli that are coprime are easiest to handle.
    # If m1, m2 are coprime, gcd=1, any r1, r2 work.
    # So a set of pairwise coprime moduli is ALWAYS valid (any residues).
    # Let's find the maximum weight independent set in a graph where edges connect non-coprime moduli?
    # No, non-coprime is still valid if residues are chosen correctly.
    # But coprime is "free".
    
    # Let's try to build a set with many coprime moduli first.
    # Then try to add others.
    
    def greedy_coprime_based():
        # Sort moduli by 1/m descending
        candidates = list(range(num_moduli))
        candidates.sort(key=lambda i: all_moduli[i]) # m ascending
        
        selected_indices = []
        # First pass: pick coprime moduli
        # Actually, just greedily pick if it doesn't conflict with existing assignment?
        # This is similar to greedy_construct.
        
        # Let's try a randomized greedy with restarts
        best_local = []
        best_local_score = 0.0
        
        for _ in range(50): # 50 restarts
            state = []
            # Randomize order slightly?
            # No, keep m ascending for value.
            for m_idx in range(num_moduli):
                m = all_moduli[m_idx]
                # Try to find a residue
                # To allow for more complex structures, we might need to backtrack in greedy?
                # Simple greedy:
                r = find_valid_residue(m_idx, state)
                if r != -1:
                    state.append((m_idx, r))
            
            score = calc_score([m_idx for m_idx, r in state])
            if score > best_local_score:
                best_local_score = score
                best_local = state[:]
            
            if time.time() > deadline - 20:
                break
        
        return best_local, best_local_score
    
    state, score = greedy_coprime_based()
    if score > best_score:
        best_state = state
        best_score = score

    # Now, try to improve by adding more elements to best_state
    # We can try to solve CSP for best_state + {new_m} for each new_m.
    # This is expensive.
    
    # Let's use a time-bounded local search on the SET.
    current_set = [m_idx for m_idx, r in best_state]
    current_sol = best_state[:]
    current_score = best_score
    
    # Pre-compute which moduli are NOT in current_set
    in_set = set(current_set)
    not_in_set = [i for i in range(num_moduli) if i not in in_set]
    
    # Sort not_in_set by 1/m descending
    not_in_set.sort(key=lambda i: all_moduli[i])
    
    # Try to add each not_in_set element, one by one
    # For each, check if current_sol + new is solvable.
    # If yes, add it, update current_sol.
    
    # This is a "Forward Checking" approach.
    # We don't change residues of existing elements, we just try to extend.
    # If we can't extend, we might need to change residues (backtracking).
    
    # Let's implement a simple local search that allows changing residues.
    # State: current_sol (list of [m_idx, r])
    # Move: 
    #   1. Pick a random unselected m_idx.
    #   2. Try to find a valid r for it given current_sol.
    #   3. If found, add it.
    #   4. If not found, try to change residues of some existing elements to make room?
    #      This is hard.
    #   5. Alternatively, try to swap: remove a random selected m_idx, then try to add the new one.
    
    # Let's try a Simulated Annealing on the RESIDUES of a fixed set of moduli?
    # No, the set changes.
    
    # Let's stick to the Set optimization.
    # Since we have ~100s, and CSP is fast for small sets,
    # we can do many "add" attempts.
    
    # Improvement: 
    # Keep a pool of "almost valid" sets?
    
    # Let's try a simple loop:
    # While time:
    #   Pick random move:
    #     1. Add random unselected (if feasible via CSP solve of size N+1)
    #     2. Swap random selected with random unselected (if feasible via CSP solve of size N)
    
    # CSP solve of size N+1 where N ~ 100 might be slow.
    # Average N in greedy is around 60-80?
    # 181 moduli.
    
    # Let's check the size of best_state.
    # If it's too big, CSP might be slow.
    
    # Optimization: Use the existing assignment as a "hint" for CSP.
    # i.e., try to keep old residues, only change if necessary.
    # This is essentially backtracking with variable ordering: new variable first?
    # Or old variables first, and try to keep their values?
    
    def solve_csp_with_hint(indices, hint_assignment, time_limit=0.5):
        # hint_assignment: dict {m_idx: r} for some indices
        # indices: list of all m_idx to assign
        # We want to find an assignment for all indices, preferring hint values.
        
        # Sort indices: those in hint first? Or new ones first?
        # If we put new ones first, we check if they fit into the existing structure easily.
        # But we also need to assign the old ones.
        
        # Let's sort by: 1. Not in hint (new/changed), 2. In hint.
        # Within groups, sort by m.
        
        def sort_key(idx):
            if idx in hint_assignment:
                return (1, all_moduli[idx])
            else:
                return (0, all_moduli[idx])
        
        sorted_indices = sorted(indices, key=sort_key)
        n = len(sorted_indices)
        assignment = [None] * n
        
        def backtrack(k):
            if time.time() > start_time + time_limit:
                return False
            if k == n:
                return True
            
            m_idx = sorted_indices[k]
            m = all_moduli[m_idx]
            
            # Order of trying residues:
            # If m_idx is in hint, try hint value first.
            if m_idx in hint_assignment:
                hint_r = hint_assignment[m_idx]
                residues_to_try = [hint_r] + [r for r in range(m) if r != hint_r]
            else:
                residues_to_try = range(m)
            
            for r in residues_to_try:
                valid = True
                for j in range(k):
                    m2_idx = sorted_indices[j]
                    r2 = assignment[j]
                    g = gcd_matrix[m_idx][m2_idx]
                    if (r - r2) % g == 0:
                        valid = False
                        break
                
                if valid:
                    assignment[k] = r
                    if backtrack(k + 1):
                        return True
                    assignment[k] = None
            
            return False

        if backtrack(0):
            return [(sorted_indices[i], assignment[i]) for i in range(n)]
        return None

    # Main Loop: Local Search on Set
    # We maintain current_set, current_sol, current_score.
    
    # To avoid expensive full CSP for every swap, we can limit the time per solve.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            # Reset to best? Or restart greedy?
            # Let's just continue, maybe randomization helps.
            no_improve_count = 0
            max_no_improve = 100 # Increase threshold
            
        # Choose move type
        # 1. Add (30%)
        # 2. Swap (50%)
        # 3. Remove (20%) - usually bad, but helps escape local optima?
        #    Actually, removing decreases score. Only useful if it allows adding 2 others?
        #    Let's skip remove for now, focus on Add/Swap.
        
        move_type = random.random()
        
        if move_type < 0.3 and not_in_set:
            # Add
            # Pick a candidate from not_in_set
            # Prefer smaller m (higher value)
            # random.choice or weighted?
            # Let's pick from the top 20 smallest m in not_in_set
            # not_in_set is already sorted by m ascending.
            # Pick randomly from first 10?
            if len(not_in_set) > 0:
                candidates = not_in_set[:10]
                add_idx = random.choice(candidates)
                
                new_set = current_set + [add_idx]
                hint = {m_idx: r for m_idx, r in current_sol}
                # Add the new one to hint? No, we want to find it.
                # But we can give it a random hint? No, leave it empty for new one.
                
                new_sol = solve_csp_with_hint(new_set, hint, time_limit=0.2)
                if new_sol:
                    current_set = new_set
                    current_sol = new_sol
                    current_score = calc_score(current_set)
                    no_improve_count = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_state = current_sol[:]
                        
                    # Update not_in_set
                    in_set.add(add_idx)
                    not_in_set.remove(add_idx)
                else:
                    no_improve_count += 1
                    
        elif move_type < 0.8:
            # Swap
            if not_in_set and current_set:
                # Pick one to remove from current_set
                # Prefer larger m (lower value) to remove
                # current_set is not sorted. Let's sort it?
                # Or just pick random.
                # To improve score, we want to remove a large m and add a small m.
                # Let's pick a random remove candidate from the larger half of current_set?
                # And a random add candidate from the smaller half of not_in_set?
                
                # Sort current_set by m descending (largest first)
                # Sort not_in_set by m ascending (smallest first)
                
                # Heuristic: Remove a "expensive" modulus, Add a "cheap" one.
                # But "cheap" (small m) has higher 1/m.
                
                # Let's try:
                # Remove candidate: random from current_set
                # Add candidate: random from not_in_set[:10]
                
                remove_idx = random.choice(current_set)
                add_candidates = not_in_set[:10]
                if not add_candidates:
                    continue
                add_idx = random.choice(add_candidates)
                
                # If add_idx == remove_idx (shouldn't happen as they are disjoint sets)
                
                new_set = [i for i in current_set if i != remove_idx] + [add_idx]
                hint = {m_idx: r for m_idx, r in current_sol if m_idx != remove_idx}
                
                new_sol = solve_csp_with_hint(new_set, hint, time_limit=0.2)
                if new_sol:
                    # Check if score improved or if we want to accept neutral/worse?
                    # For now, only accept if score >= current?
                    # Since swap keeps size same, score changes.
                    new_score = calc_score(new_set)
                    
                    # Accept if new_score >= current_score (or with prob for SA)
                    # Let's be greedy: accept if new_score > current_score
                    if new_score > current_score:
                        current_set = new_set
                        current_sol = new_sol
                        current_score = new_score
                        no_improve_count = 0
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_state = current_sol[:]
                            
                        # Update sets
                        in_set.remove(remove_idx)
                        not_in_set.remove(add_idx)
                        in_set.add(add_idx)
                        not_in_set.append(remove_idx)
                        # Re-sort not_in_set?
                        # not_in_set.sort(key=lambda i: all_moduli[i]) # Expensive
                        # Let's just append and sort occasionally
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        else:
            # Perturb: Randomly change a few residues?
            # Doesn't change score, but might help future adds.
            # Skip for now.
            pass

    # Final check
    if best_score > 0:
        result = []
        for m_idx, r in best_state:
            result.append([all_moduli[m_idx], r])
        return result
    else:
        # Fallback
        return [[20, 0]]

# EVOLVE-BLOCK-END
