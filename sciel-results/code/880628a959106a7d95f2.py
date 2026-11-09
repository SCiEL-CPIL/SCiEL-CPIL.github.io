"""Maximize sum of 1/m for pairwise disjoint residue classes."""
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # CSP Solver with Forward Checking and MRV
    def is_feasible(indices):
        """Check if a set of moduli indices has a valid residue assignment."""
        if not indices:
            return True
        
        # Sort by modulus size (smallest first) to help MRV
        indices = sorted(indices, key=lambda i: moduli[i])
        
        # Domains: list of sets
        domains = [set(range(moduli[i])) for i in indices]
        
        def backtrack(k):
            if k == len(indices):
                return True
            
            # MRV Heuristic: Find variable with smallest domain
            min_size = float('inf')
            min_pos = -1
            for p in range(k, len(indices)):
                size = len(domains[p])
                if size < min_size:
                    min_size = size
                    min_pos = p
                if size == 0:
                    return False
            
            if min_pos == -1:
                return True

            # Swap to bring min_pos to k
            indices[k], indices[min_pos] = indices[min_pos], indices[k]
            domains[k], domains[min_pos] = domains[min_pos], domains[k]
            
            idx_k = indices[k]
            
            # Try each value in the domain
            # Shuffle for diversity in local search
            vals = list(domains[k])
            random.shuffle(vals)
            
            for r in vals:
                # Propagate: Assign r to idx_k
                old_domain_k = domains[k]
                domains[k] = {r}
                
                # Forward Check: Update neighbors
                consistent = True
                changes = []
                
                for p in range(k + 1, len(indices)):
                    idx_p = indices[p]
                    g = gcd_table[idx_k][idx_p]
                    
                    if g == 1:
                        continue
                    
                    old_dom_p = domains[p]
                    new_dom_p = set()
                    for r_p in old_dom_p:
                        if (r_p - r) % g != 0:
                            new_dom_p.add(r_p)
                    
                    if not new_dom_p:
                        consistent = False
                        break
                    
                    changes.append((p, old_dom_p, new_dom_p))
                    domains[p] = new_dom_p
                
                if consistent:
                    if backtrack(k + 1):
                        return True
                
                # Undo changes
                for p, old_dom, _ in changes:
                    domains[p] = old_dom
                domains[k] = old_domain_k
            
            # Undo swap
            indices[k], indices[min_pos] = indices[min_pos], indices[k]
            domains[k], domains[min_pos] = domains[min_pos], domains[k]
            
            return False

        return backtrack(0)

    def get_score(indices):
        return sum(1.0 / moduli[i] for i in indices)

    # Greedy Initialization
    def greedy_init():
        selected_indices = []
        # Sort by 1/m descending (small m first)
        sorted_mods = sorted(range(n_mod), key=lambda i: 1.0/moduli[i])
        
        for i in sorted_mods:
            # Try to add i
            # Check feasibility of selected_indices + [i]
            # This is expensive if done naively.
            # Instead, try to find a residue manually?
            # No, just use is_feasible.
            # But is_feasible is slow.
            
            # Heuristic: Only add if the new set is feasible.
            # To speed up, we can check if the new modulus conflicts with existing ones
            # in a way that makes it impossible?
            
            # Let's just try to add and check.
            # If it fails, skip.
            
            # Optimization: If the set is large, is_feasible is slow.
            # We can limit the size of the greedy set?
            # No, we want max score.
            
            # Try adding
            new_set = selected_indices + [i]
            if is_feasible(new_set):
                selected_indices = new_set
            
            if time.time() > deadline - 10:
                break
                
        return selected_indices

    # Local Search (Simulated Annealing)
    def local_search(initial_indices, max_time):
        current_indices = list(initial_indices)
        current_score = get_score(current_indices)
        
        best_indices = list(current_indices)
        best_score = current_score
        
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        
        step = 0
        while time.time() < deadline - 2:
            step += 1
            
            # Pick a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_indices = list(current_indices)
            
            if move_type == 'add':
                # Pick a random modulus not in current set
                missing = [i for i in range(n_mod) if i not in current_indices]
                if missing:
                    i_add = random.choice(missing)
                    new_indices.append(i_add)
            
            elif move_type == 'remove':
                if current_indices:
                    i_rem = random.choice(current_indices)
                    new_indices.remove(i_rem)
            
            elif move_type == 'swap':
                if current_indices:
                    i_rem = random.choice(current_indices)
                    missing = [i for i in range(n_mod) if i not in current_indices]
                    if missing:
                        i_add = random.choice(missing)
                        new_indices.remove(i_rem)
                        new_indices.append(i_add)
            
            # Check feasibility
            if is_feasible(new_indices):
                new_score = get_score(new_indices)
                delta = new_score - current_score
                
                # Accept if better, or with probability exp(delta/T)
                if delta > 0 or random.random() < pow(2.71828, delta / T):
                    current_indices = new_indices
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_indices = list(current_indices)
            
            T *= cooling
            
            if T < T_min:
                T = T_min
                # Reheat occasionally?
                if step % 1000 == 0:
                    T = 0.005

        return best_indices, best_score

    # Main Logic
    # 1. Greedy Init
    initial_indices = greedy_init()
    
    # 2. Local Search
    best_indices, best_score = local_search(initial_indices, deadline - time.time())
    
    # 3. Return the best set of [m, r]
    # We need to find the actual residues for the best set.
    # We can run the CSP solver again to get the residues.
    
    # Modify is_feasible to return residues?
    # Or just run a simple backtracking to find one solution.
    
    def find_residues(indices):
        if not indices:
            return {}
        
        indices = sorted(indices, key=lambda i: moduli[i])
        assigned = {}
        
        def backtrack(k):
            if k == len(indices):
                return True
            
            idx = indices[k]
            m = moduli[idx]
            
            for r in range(m):
                ok = True
                for j in range(k):
                    idx_j = indices[j]
                    r_j = assigned[idx_j]
                    g = gcd_table[idx][idx_j]
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    assigned[idx] = r
                    if backtrack(k + 1):
                        return True
                    del assigned[idx]
            return False
        
        if backtrack(0):
            return assigned
        return {}

    residues = find_residues(best_indices)
    
    result = []
    for i in best_indices:
        m = moduli[i]
        r = residues.get(i, 0)
        result.append([m, r])
    
    return result

# EVOLVE-BLOCK-END
