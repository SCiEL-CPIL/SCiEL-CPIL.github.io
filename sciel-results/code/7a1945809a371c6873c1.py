"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MAX_MOD = 128
    
    # Precompute gcd for all pairs
    moduli_list = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli_list)
    
    # gcd matrix
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli_list[i], moduli_list[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # For a chosen set of moduli, we need to assign residues r_i in [0, m_i)
    # such that for all i<j: gcd(m_i, m_j) does NOT divide (r_i - r_j)
    
    # Greedy approach: try to add moduli one by one, choosing residue that works
    # with all previously chosen ones.
    
    def can_assign_residue(mod_idx, chosen_mods, chosen_res):
        """Try to find a residue for mod_idx that's disjoint from all chosen."""
        m = moduli_list[mod_idx]
        for r in range(m):
            ok = True
            for k in range(len(chosen_mods)):
                g = gcd_mat[mod_idx][chosen_mods[k]]
                diff = (r - chosen_res[k]) % g
                if diff == 0:
                    ok = False
                    break
            if ok:
                return r
        return -1
    
    # Strategy: try different orderings of moduli, greedy assign
    # Use multiple random restarts with different orderings
    
    import random
    random.seed(42)
    
    best_count = 0
    best_result = None
    
    # Try several heuristics for ordering
    # 1. Descending by number of divisors (more constrained first?)
    # 2. Random orderings
    
    def count_divisors(n):
        c = 0
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                c += 2
                if i * i == n:
                    c -= 1
        return c + 1
    
    div_counts = [count_divisors(m) for m in moduli_list]
    
    # Order by descending divisor count (more composite first, more shared factors)
    order_by_div = sorted(range(n_mods), key=lambda i: -div_counts[i])
    
    # Also try ascending modulus (smaller moduli have fewer residue options)
    order_by_mod = sorted(range(n_mods), key=lambda i: moduli_list[i])
    
    orders_to_try = [order_by_div, order_by_mod]
    
    # Add some random orders
    for _ in range(20):
        random_order = list(range(n_mods))
        random.shuffle(random_order)
        orders_to_try.append(random_order)
    
    for order in orders_to_try:
        if time.time() > deadline:
            break
        
        chosen_mods = []
        chosen_res = []
        
        for mod_idx in order:
            r = can_assign_residue(mod_idx, chosen_mods, chosen_res)
            if r >= 0:
                chosen_mods.append(mod_idx)
                chosen_res.append(r)
        
        count = len(chosen_mods)
        if count > best_count:
            best_count = count
            best_result = [[chosen_res[i], moduli_list[chosen_mods[i]]] for i in range(count)]
        
        # Early stop if we got a perfect solution (all moduli)
        if best_count == n_mods:
            break
    
    if best_result is None:
        # Fallback: at least return something valid
        # Try just moduli that are powers of 2 with residue 1
        result = []
        for k in range(1, 8):
            m = 1 << k
            if m <= MAX_MOD:
                result.append([1, m])
        return result
    
    return best_result
# EVOLVE-BLOCK-END
