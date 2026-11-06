import time
from math import gcd

_START = time.time()
_DEADLINE = 140.0

def construct():
        chosen = []
        
        def is_compatible(m, r, chosen_list):
            for mk, rk in chosen_list:
                g = gcd(m, mk)
                if (r % g) == (rk % g):
                    return False
            return True

        def find_valid_residues(m, chosen_list):
            valid = []
            for r in range(m):
                if is_compatible(m, r, chosen_list):
                    valid.append(r)
            return valid

        # Greedy Phase
        for m in range(20, 201):
            if time.time() - _START > _DEADLINE:
                break
                
            valid_residues = find_valid_residues(m, chosen)
            if not valid_residues:
                continue
                
            if len(valid_residues) == 1:
                chosen.append([m, valid_residues[0]])
            else:
                best_r = valid_residues[0]
                max_potential = -1
                
                for r in valid_residues:
                    # Potential: Sum of 1/k for k > m that have at least one valid residue
                    # given (m, r) is added.
                    potential = 0.0
                    temp_chosen = chosen + [[m, r]]
                    
                    for k in range(m + 1, 201):
                        # Check if k has any valid residue
                        # Optimization: Just check a few? Or all?
                        # m is up to 200. k is up to 200.
                        # Number of k's is ~180.
                        # For each k, we check up to k residues.
                        # This is O(M^3) roughly. 200^3 = 8e6. Feasible.
                        
                        # Fast check: Does k have ANY valid residue?
                        # We can stop early if we find one.
                        has_valid = False
                        for rk in range(k):
                            if is_compatible(k, rk, temp_chosen):
                                has_valid = True
                                break
                                
                        if has_valid:
                            potential += 1.0 / k
                            
                    if potential > max_potential:
                        max_potential = potential
                        best_r = r
                        # Early break if potential is very high?
                        
                chosen.append([m, best_r])
                
        # Local Improvement Phase
        # Try to improve by swapping residues or adding skipped moduli.
        # Since we iterated 20..200, skipped moduli are those with no valid residue.
        # Changing a residue of an existing modulus might free up space for a skipped one.
        
        improved = True
        while improved and (time.time() - _START < _DEADLINE):
            improved = False
            # Identify skipped moduli
            chosen_moduli = set(m for m, r in chosen)
            skipped_moduli = [m for m in range(20, 201) if m not in chosen_moduli]
            
            # For each existing modulus, try changing its residue
            for i, (m, r) in enumerate(chosen):
                if time.time() - _START > _DEADLINE:
                    break
                    
                # Find all valid residues for m, excluding current r?
                # Or just try all valid residues.
                valid_residues = find_valid_residues(m, [c for j, c in enumerate(chosen) if j != i])
                
                for new_r in valid_residues:
                    if new_r == r:
                        continue
                    
                    # Tentatively change
                    old_r = chosen[i][1]
                    chosen[i][1] = new_r
                    
                    # Check if any skipped modulus can now be added
                    added = False
                    for k in skipped_moduli:
                        if is_compatible(k, 0, chosen): # Check if k has ANY valid residue
                             # Actually, we need to find a valid residue for k
                             valid_k = find_valid_residues(k, chosen)
                             if valid_k:
                                 # Add k
                                 chosen.append([k, valid_k[0]])
                                 added = True
                                 # Update skipped list?
                                 # Break out of k loop, try next m
                                 break
                     
                    if added:
                        improved = True
                        # Don't break m loop, try to add more?
                        # Or break to restart?
                        # Let's break to restart the while loop to re-evaluate.
                        break
                    else:
                        # Revert
                        chosen[i][1] = old_r
                        
                if improved:
                    break
                    
        return chosen
