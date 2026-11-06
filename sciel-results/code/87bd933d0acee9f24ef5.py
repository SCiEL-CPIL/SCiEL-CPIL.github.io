import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    moduli = list(range(2, 129))
    best_solution = []
    best_count = 0
    
    # Precompute gcds
    gcds = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            gcds[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        return gcds.get((m1, m2), 1)

    def is_compatible(residue, modulus, assigned):
        for r, m in assigned:
            g = get_gcd(modulus, m)
            if residue % g == r % g:
                return False
        return True

    def backtrack(assigned, unassigned):
        nonlocal best_solution, best_count
        if time.time() > deadline:
            return
            
        if len(assigned) > best_count:
            best_count = len(assigned)
            best_solution = [x[:] for x in assigned]
            
        if not unassigned:
            return
            
        # MRV: Pick variable (modulus) with fewest valid residues
        min_valid = float('inf')
        best_var_idx = -1
        best_var_residues = []
        
        for idx, m in enumerate(unassigned):
            valid_residues = []
            for r in range(m):
                if is_compatible(r, m, assigned):
                    valid_residues.append(r)
            if len(valid_residues) < min_valid:
                min_valid = len(valid_residues)
                best_var_idx = idx
                best_var_residues = valid_residues
                if min_valid == 0:
                    break
                if min_valid == 1:
                    # Can't get better than 1 (except 0 which we break on)
                    # But we should check all to be sure? No, 1 is min possible non-zero.
                    # Actually, if we find 1, we can stop checking others for this step?
                    # No, another variable might have 0.
                    # So we must check all if min_valid > 0.
                    pass
        
        if min_valid == 0:
            return
            
        m = unassigned[best_var_idx]
        new_unassigned = unassigned[:best_var_idx] + unassigned[best_var_idx+1:]
        
        # Shuffle residues to randomize
        random.shuffle(best_var_residues)
        
        for r in best_var_residues:
            assigned.append([r, m])
            backtrack(assigned, new_unassigned)
            assigned.pop()
            
            if time.time() > deadline:
                return

    # Run a few restarts
    for _ in range(5):
        random.shuffle(moduli)
        assigned = []
        backtrack(assigned, moduli[:])
        if time.time() > deadline:
            break
        if best_count == len(moduli):
            break

    return best_solution
