"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_moduli = len(moduli)
    
    # Precompute gcd table
    gcd_table = np.zeros((n_moduli, n_moduli), dtype=np.int32)
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i, j] = g
            gcd_table[j, i] = g
    
    # Precompute conflict masks:
    # conflict_mask[i, j, r_i] = bitset of residues r_j that conflict with r_i
    # But this is huge. Instead, for each pair (i,j), precompute for each possible
    # value v in [0, g-1] where g=gcd(i,j), the mask of residues of j with r_j % g == v.
    
    # For each modulus j, precompute residue_classes[j][g][v] = bitset of r in [0,m_j) with r%g==v
    # But g varies per pair. Let's compute on the fly but use numpy for speed.
    
    # Alternative: for each modulus i, precompute for each other modulus j,
    # a 2D array: compat[i][j][r_i] = boolean array of length m_j, True if compatible
    
    # This is n_moduli^2 * max_m booleans = 127^2 * 128 = ~2M booleans = 2MB. OK.
    
    # Actually, let's precompute for each pair (i,j) and each r_i, the forbidden residues for j.
    # Use a list of bytearrays.
    
    # For speed, let's precompute: for each i, j, and r_i, a numpy bool array of length m_j
    # indicating which r_j are compatible.
    
    # Memory: sum over i,j of m_i * m_j booleans. This is sum_i m_i * sum_j m_j = (sum m)^2 / 4 roughly.
    # sum m from 2 to 128 = 128*129/2 - 1 = 8255. So 8255^2 = ~68M booleans = 68MB. Too much.
    
    # Better: for each pair (i,j), precompute a 2D array of shape (m_i, m_j) with True if compatible.
    # Total: sum_{i<j} m_i * m_j. This is (8255^2 - sum m_i^2)/2. sum m_i^2 = sum_{m=2}^{128} m^2 = 128*129*257/6 - 1 = 708624 - 1 = 708623.
    # (8255^2 - 708623)/2 = (68145025 - 708623)/2 = 67436402/2 = 33718201 booleans = ~34MB. Acceptable.
    
    # But building this takes time. Let's compute on the fly with numpy vectorization.
    
    def get_compatible_mask(i, j, r_i):
        """Return numpy bool array of length m_j, True where compatible with r_i at modulus i."""
        m_j = moduli[j]
        g = gcd_table[i, j]
        # Compatible if r_j % g != r_i % g
        target = r_i % g
        r_j = np.arange(m_j)
        return (r_j % g) != target
    
    # This is called many times. Let's cache.
    cache = {}
    
    def get_compatible_mask_cached(i, j, r_i):
        key = (i, j, r_i)
        if key not in cache:
            cache[key] = get_compatible_mask(i, j, r_i)
        return cache[key]
    
    # For greedy: for modulus i, given current assignment, find first compatible residue.
    # For each candidate r, check against all assigned moduli.
    # Use numpy for the check.
    
    def find_compatible_residue(i, assignment):
        """Find a residue for modulus i compatible with all in assignment. Return r or -1."""
        m_i = moduli[i]
        # Start with all residues allowed
        allowed = np.ones(m_i, dtype=bool)
        
        for j, r_j in assignment.items():
            if j == i:
                continue
            g = gcd_table[i, j]
            target = r_j % g
            # Forbid residues r where r % g == target
            r_i = np.arange(m_i)
            allowed &= (r_i % g) != target
        
        if not np.any(allowed):
            return -1
        return int(np.argmax(allowed))
    
    def greedy_assign(order):
        assignment = {}
        for i in order:
            r = find_compatible_residue(i, assignment)
            if r >= 0:
                assignment[i] = r
        return assignment
    
    def local_search_improve(assignment, max_time=5.0):
        ls_start = time.time()
        assigned = set(assignment.keys())
        
        while time.time() - ls_start < max_time:
            improved = False
            
            # Try adding unassigned moduli
            for i in range(n_moduli):
                if i in assigned:
                    continue
                r = find_compatible_residue(i, assignment)
                if r >= 0:
                    assignment[i] = r
                    assigned.add(i)
                    improved = True
            
            if improved:
                continue
            
            # Try changing residues of assigned moduli
            assigned_list = list(assigned)
            random.shuffle(assigned_list)
            for i in assigned_list:
                m_i = moduli[i]
                current_r = assignment[i]
                # Try all other residues
                for r in range(m_i):
                    if r == current_r:
                        continue
                    # Check if r is compatible with all others
                    ok = True
                    for j, r_j in assignment.items():
                        if j == i:
                            continue
                        g = gcd_table[i, j]
                        if r % g == r_j % g:
                            ok = False
                            break
                    if ok:
                        assignment[i] = r
                        # Try to add more
                        for j in range(n_moduli):
                            if j in assigned:
                                continue
                            rj = find_compatible_residue(j, assignment)
                            if rj >= 0:
                                assignment[j] = rj
                                assigned.add(j)
                        improved = True
                        break
            
            if not improved:
                break
        
        return assignment
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    best_result = None
    best_count = 0
    
    # Generate orderings
    orderings = []
    orderings.append(list(range(n_moduli)))
    orderings.append(list(range(n_moduli - 1, -1, -1)))
    orderings.append(sorted(range(n_moduli), key=lambda i: (not is_prime(moduli[i]), moduli[i])))
    orderings.append(sorted(range(n_moduli), key=lambda i: (is_prime(moduli[i]), moduli[i])))
    
    rng = random.Random(12345)
    for _ in range(20):
        order = list(range(n_moduli))
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        assignment = greedy_assign(order)
        count = len(assignment)
        if count > best_count:
            best_count = count
            best_result = dict(assignment)
        
        remaining = deadline - time.time()
        if remaining > 2.0:
            assignment = local_search_improve(assignment, max_time=min(3.0, remaining - 1.0))
            count = len(assignment)
            if count > best_count:
                best_count = count
                best_result = dict(assignment)
    
    if best_result is None or len(best_result) == 0:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    result = [[best_result[i], moduli[i]] for i in sorted(best_result.keys())]
    return result

# EVOLVE-BLOCK-END
