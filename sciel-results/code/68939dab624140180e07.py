import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Candidate moduli
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
            
    # Precompute 1/m
    inv_m = [1.0 / m for m in moduli]
    
    def get_score(selected):
        return sum(1.0 / m for m, r in selected)
        
    def is_compatible(idx_m, r, selected_idx_res):
        """
        Check if (idx_m, r) is compatible with all entries in selected_idx_res.
        selected_idx_res: list of (idx_prime, r_prime)
        """
        for idx_prime, r_prime in selected_idx_res:
            g = gcd_mat[idx_m][idx_prime]
            if r % g == r_prime % g:
                return False
        return True

    def find_valid_residue(idx_m, selected_idx_res):
        """
        Find a valid residue for idx_m given selected_idx_res.
        Returns None if not found.
        """
        m = moduli[idx_m]
        # Try residues in a shuffled order to avoid bias
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible(idx_m, r, selected_idx_res):
                return r
        return None

    def greedy_fill(seed, priority_order=None):
        random.seed(seed)
        selected_idx_res = []
        selected_set = set()
        
        if priority_order is None:
            # Default: smaller moduli first (higher weight)
            order = list(range(n_mods))
        else:
            order = priority_order
            
        for idx_m in order:
            r = find_valid_residue(idx_m, selected_idx_res)
            if r is not None:
                selected_idx_res.append((idx_m, r))
                selected_set.add(idx_m)
                
        result = []
        for idx_m, r in selected_idx_res:
            result.append([moduli[idx_m], r])
        return result

    def local_search(solution, seed, max_iters=2000):
        random.seed(seed)
        
        # Build index mapping: idx -> (m, r)
        sol_dict = {}
        for m, r in solution:
            idx = m - 20
            sol_dict[idx] = r
        
        current_score = get_score(solution)
        best_score = current_score
        best_sol = list(solution)
        
        # Convert best_sol to idx_res format for fast ops?
        # We will maintain sol_dict and best_sol in sync.
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            move_type = random.choice(['swap', 'insert', 'delete_insert'])
            
            if move_type == 'swap':
                # Try to change residue of an existing modulus to enable future inserts
                # Or just randomize to escape local minima
                if not sol_dict:
                    continue
                idx_m = random.choice(list(sol_dict.keys()))
                m = moduli[idx_m]
                r_old = sol_dict[idx_m]
                
                # Try a few random new residues
                for _ in range(5):
                    r_new = random.randint(0, m - 1)
                    if r_new == r_old:
                        continue
                    
                    # Check compatibility with others
                    compatible = True
                    for idx_prime, r_prime in sol_dict.items():
                        if idx_prime == idx_m:
                            continue
                        g = gcd_mat[idx_m][idx_prime]
                        if r_new % g == r_prime % g:
                            compatible = False
                            break
                    
                    if compatible:
                        # Accept swap (score doesn't change, but state does)
                        sol_dict[idx_m] = r_new
                        # Update best_sol? No, best_sol is for max score.
                        # We only update best_sol if score improves.
                        # But this swap might enable an insert later.
                        pass
            
            elif move_type == 'insert':
                # Try to insert a missing modulus
                missing = [i for i in range(n_mods) if i not in sol_dict]
                if not missing:
                    continue
                
                # Prioritize smaller moduli
                missing.sort(key=lambda i: inv_m[i], reverse=True)
                # Pick from top 5
                candidates = missing[:5]
                idx_m = random.choice(candidates)
                m = moduli[idx_m]
                
                r = find_valid_residue(idx_m, [(i, sol_dict[i]) for i in sol_dict])
                if r is not None:
                    # Insert
                    sol_dict[idx_m] = r
                    # Rebuild solution list for score check
                    new_solution = [[moduli[i], sol_dict[i]] for i in sol_dict]
                    new_score = get_score(new_solution)
                    if new_score > best_score + 1e-15:
                        best_score = new_score
                        best_sol = new_solution
            
            elif move_type == 'delete_insert':
                # Remove a large modulus, try to insert a smaller one
                if len(sol_dict) <= 1:
                    continue
                
                # Pick a candidate to delete (prefer large m)
                items = list(sol_dict.items())
                items.sort(key=lambda x: inv_m[x[0]]) # Ascending 1/m -> Large m first
                # Pick from top 5 largest m
                del_candidates = [i for i, _ in items[:5]]
                idx_del = random.choice(del_candidates)
                
                # Save state
                r_del = sol_dict[idx_del]
                del sol_dict[idx_del]
                
                # Try to insert a smaller modulus
                missing = [i for i in range(n_mods) if i not in sol_dict]
                if missing:
                    missing.sort(key=lambda i: inv_m[i], reverse=True)
                    ins_candidates = missing[:5]
                    idx_ins = random.choice(ins_candidates)
                    
                    r_ins = find_valid_residue(idx_ins, [(i, sol_dict[i]) for i in sol_dict])
                    if r_ins is not None:
                        sol_dict[idx_ins] = r_ins
                        new_solution = [[moduli[i], sol_dict[i]] for i in sol_dict]
                        new_score = get_score(new_solution)
                        
                        if new_score > best_score + 1e-15:
                            best_score = new_score
                            best_sol = new_solution
                        else:
                            # Revert
                            del sol_dict[idx_ins]
                            sol_dict[idx_del] = r_del
                    else:
                        # Revert
                        sol_dict[idx_del] = r_del
                else:
                    # No missing, revert
                    sol_dict[idx_del] = r_del
                    
        return best_sol

    best_solution = []
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        seed = num_restarts + 1
        
        # Create priority order: biased towards small moduli
        indices = list(range(n_mods))
        if random.random() < 0.7:
            # Sort by 1/m descending (small m first)
            indices.sort(key=lambda i: -inv_m[i])
        else:
            random.shuffle(indices)
            
        sol = greedy_fill(seed, priority_order=indices)
        score = get_score(sol)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
        # Local search on this solution
        if time.time() < deadline - 15.0:
            improved_sol = local_search(sol, seed + 1000, max_iters=1000)
            improved_score = get_score(improved_sol)
            if improved_score > best_score:
                best_score = improved_score
                best_solution = improved_sol
                
        num_restarts += 1
        
    return best_solution
