"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute all 4-subsets of [1,16] as bitmasks (16-bit ints)
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_candidates = len(all_sets)  # 1820

    # Precompute pairwise intersection sizes (popcount of bitwise AND)
    # dist[i][j] = |all_sets[i] ∩ all_sets[j]|
    dist = [[0] * n_candidates for _ in range(n_candidates)]
    for i in range(n_candidates):
        ai = all_sets[i]
        for j in range(i + 1, n_candidates):
            d = (ai & all_sets[j]).bit_count()
            dist[i][j] = d
            dist[j][i] = d

    def check_conflict(family_indices, new_idx):
        """Check if adding new_idx to family creates a sunflower triple.
        A sunflower triple {A,B,C} has A∩B = A∩C = B∩C.
        """
        n_fam = len(family_indices)
        for a in range(n_fam):
            i = family_indices[a]
            dik = dist[i][new_idx]
            for b in range(a + 1, n_fam):
                j = family_indices[b]
                # Check if |A∩B| == |A∩C| == |B∩C|
                if dist[i][j] == dik == dist[j][new_idx]:
                    return True
        return False

    def greedy_run(order):
        """Run greedy algorithm with given ordering of candidates."""
        family = []
        for idx in order:
            if not check_conflict(family, idx):
                family.append(idx)
        return family

    def local_search(family, max_iterations=1000):
        """Improve family via local search: try swapping or adding elements."""
        family = family[:]
        in_family = set(family)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() - start_time < DEADLINE:
            improved = False
            iterations += 1
            
            # Try to add any candidate not in family
            for idx in range(n_candidates):
                if idx in in_family:
                    continue
                if not check_conflict(family, idx):
                    family.append(idx)
                    in_family.add(idx)
                    improved = True
                    break
            
            # If no addition possible, try swaps
            if not improved:
                for remove_idx in family[:]:
                    temp_fam = [x for x in family if x != remove_idx]
                    temp_in = set(temp_fam)
                    for add_idx in range(n_candidates):
                        if add_idx in temp_in:
                            continue
                        if not check_conflict(temp_fam, add_idx):
                            # Check if this swap is beneficial (it keeps size same, 
                            # but may enable future additions)
                            # For now, accept if it doesn't create conflict
                            family = temp_fam + [add_idx]
                            in_family = temp_in | {add_idx}
                            improved = True
                            break
                    if improved:
                        break
        
        return family

    # Try multiple random orderings with greedy + local search
    random.seed(42)
    best_family = []
    best_size = 0
    
    # First, try a few deterministic orderings
    deterministic_orders = [
        list(range(n_candidates)),
        list(range(n_candidates - 1, -1, -1)),
    ]
    
    for order in deterministic_orders:
        if time.time() - start_time > DEADLINE - 10:
            break
        family = greedy_run(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Then try random orderings
    num_random_trials = 0
    while time.time() - start_time < DEADLINE - 5:
        num_random_trials += 1
        order = list(range(n_candidates))
        random.shuffle(order)
        
        family = greedy_run(order)
        
        # Apply local search if we have time
        if time.time() - start_time < DEADLINE - 2:
            family = local_search(family, max_iterations=200)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
