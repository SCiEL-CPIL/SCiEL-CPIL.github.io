import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 90.0
    
    # Moduli 2 to 128
    moduli = list(range(2, 129))
    
    # Precompute gcds
    gcd_matrix = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    # Helper to check if a new modulus can be added to a set
    # This is just gcd > 1 check for clique property
    def is_clique_candidate(m, current_set):
        for m2 in current_set:
            if math.gcd(m, m2) == 1:
                return False
        return True

    # Helper to check if residues can be assigned
    # Backtracking on residues
    def check_residues(mod_set):
        # mod_set is a list of moduli
        # Assign r_m for each m
        # Constraint: r_i != r_j mod gcd(m_i, m_j)
        
        # Order moduli by smallest number of constraints? 
        # Just use fixed order for now
        n = len(mod_set)
        assignments = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            m = mod_set[idx]
            # Try all residues
            for r in range(m):
                conflict = False
                for k in range(idx):
                    m2 = mod_set[k]
                    g = math.gcd(m, m2)
                    if r % g == assignments[k] % g:
                        conflict = True
                        break
                if not conflict:
                    assignments[idx] = r
                    if backtrack(idx + 1):
                        return True
            assignments[idx] = None
            return False
        
        return backtrack(0)

    # Greedy to get initial solution
    # Try to build a set by adding moduli that keep it a clique and check residues
    # This is expensive to check residues at every step.
    # Instead, find a large clique first, then check residues.
    
    # Find max clique in gcd>1 graph
    # Use Bron-Kerbosch or simple B&B
    # Given 127 nodes, B&B is needed.
    
    # Sort nodes by degree (number of neighbors) to prune better
    degrees = [0] * len(moduli)
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            if math.gcd(moduli[i], moduli[j]) > 1:
                degrees[i] += 1
                degrees[j] += 1
                
    # Order indices by degree descending
    order = sorted(range(len(moduli)), key=lambda i: degrees[i], reverse=True)
    ordered_moduli = [moduli[i] for i in order]
    
    best_size = 0
    best_set = []
    
    # B&B
    def solve_bnb(idx, current_set, available_indices):
        nonlocal best_size, best_set
        if time.time() > deadline - 5:
            return
        
        # Pruning: if len(current_set) + len(available_indices) <= best_size, return
        if len(current_set) + len(available_indices) <= best_size:
            return
            
        if idx == len(ordered_moduli):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        m = ordered_moduli[idx]
        # Try including m
        if is_clique_candidate(m, current_set):
            # Check if we can add m
            new_set = current_set + [m]
            # Check residues? 
            # Only check if size is close to best or if we are at a leaf?
            # Checking residues at every node is too slow.
            # We will only check residues when we update best_size or at the end.
            # But we might waste time on large cliques that are unsatisfiable.
            # Heuristic: Check residues if len(new_set) >= best_size.
            if len(new_set) >= best_size:
                if check_residues(new_set):
                    best_size = len(new_set)
                    best_set = new_set[:]
                    # Try to expand?
            
            # Recurse
            # Update available indices: remove those that conflict with m? 
            # No, we just skip them in the next steps if they are not compatible.
            # Since we process in order, we just pass idx+1.
            solve_bnb(idx + 1, new_set, available_indices[:idx] if False else None) # Pseudo code
            
        # Try excluding m
        solve_bnb(idx + 1, current_set, available_indices)

    # The above B&B structure is flawed for available_indices.
    # Standard B&B:
    # 1. Global list of moduli.
    # 2. Index pointer.
    # 3. Current set.
    
    def solve_bnb_v2(idx, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline - 5:
            return
        
        if len(current_set) + (len(ordered_moduli) - idx) <= best_size:
            return
        
        if idx == len(ordered_moduli):
            if len(current_set) > best_size:
                if check_residues(current_set):
                    best_size = len(current_set)
                    best_set = current_set[:]
            return
        
        m = ordered_moduli[idx]
        
        # Option 1: Include m
        if is_clique_candidate(m, current_set):
            new_set = current_set + [m]
            # Heuristic: if this set is larger than best, check residues
            if len(new_set) > best_size:
                if check_residues(new_set):
                    best_size = len(new_set)
                    best_set = new_set[:]
            # Recurse
            solve_bnb_v2(idx + 1, new_set)
            
        # Option 2: Exclude m
        solve_bnb_v2(idx + 1, current_set)

    solve_bnb_v2(0, [])
    
    if not best_set:
        # Fallback
        return [[0, 2], [1, 4]]
        
    # Find residues for best_set
    # We know it's valid from check_residues, but we need the actual values
    # Re-run a simple assignment
    # Since check_residues only returned bool, we need to capture the assignment.
    # Let's modify check_residues to return the assignment or run a separate one.
    
    # For simplicity, if best_set is small, we can just return it and assume the verifier
    # will check. But we need to return valid pairs.
    # The task says "return ... pairs". It implies we must provide valid r.
    # My check_residues confirms existence. I need to extract it.
    
    # Re-implement a solver that returns the assignment
    def get_residues(mod_set):
        n = len(mod_set)
        assignments = [None] * n
        def backtrack(idx):
            if idx == n:
                return True
            m = mod_set[idx]
            for r in range(m):
                conflict = False
                for k in range(idx):
                    m2 = mod_set[k]
                    g = math.gcd(m, m2)
                    if r % g == assignments[k] % g:
                        conflict = True
                        break
                if not conflict:
                    assignments[idx] = r
                    if backtrack(idx + 1):
                        return True
            assignments[idx] = None
            return False
        if backtrack(0):
            return [[assignments[i], mod_set[i]] for i in range(n)]
        return None

    result = get_residues(best_set)
    if result is None:
        return [[0, 2], [1, 4]]
    return result
