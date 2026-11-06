"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s safety margin before 60s limit
    
    # Generate all 4-element subsets of [1,16] as bitmasks
    all_masks = []
    mask_to_idx = {}
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        mask_to_idx[mask] = len(all_masks)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    def is_sunflower(a, b, c):
        # Check if three sets form a sunflower (all pairwise intersections equal)
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def greedy_with_ordering(order):
        chosen = []
        chosen_set = set()
        for idx in order:
            m = all_masks[idx]
            conflict = False
            for i, a in enumerate(chosen):
                for b in chosen[i+1:]:
                    if is_sunflower(a, b, m):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                chosen.append(m)
                chosen_set.add(idx)
        return chosen
    
    def local_search(family, max_iterations=5000):
        """Try to improve family by swapping out members that cause conflicts."""
        family_set = set(family)
        current_size = len(family)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            
            # Try adding random unchosen sets
            unchosen = [i for i in range(n_total) if i not in family_set]
            if not unchosen:
                break
            
            random.shuffle(unchosen)
            for idx in unchosen[:100]:  # Try up to 100 candidates
                m = all_masks[idx]
                # Check if adding m causes any sunflower with existing pairs
                conflict_pairs = []
                for i, a in enumerate(family):
                    for b in family[i+1:]:
                        if is_sunflower(a, b, m):
                            conflict_pairs.append((i, family.index(b)))
                
                if not conflict_pairs:
                    family.append(m)
                    family_set.add(idx)
                    improved = True
                    break
                else:
                    # Try removing one of the conflicting sets and adding m
                    for i, j in conflict_pairs[:5]:
                        old_a = family[i]
                        old_b = family[j]
                        new_family = [x for k, x in enumerate(family) if k != i and k != j]
                        new_family.append(m)
                        
                        # Verify new family is valid
                        valid = True
                        for p, a in enumerate(new_family):
                            for b in new_family[p+1:]:
                                for c in new_family[p+2:]:
                                    if is_sunflower(a, b, c):
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            family = new_family
                            family_set = set(family)
                            improved = True
                            break
                    if improved:
                        break
        
        return family
    
    # Strategy 1: Multiple random greedy with different orderings
    best_family = []
    best_size = 0
    
    for seed in range(30):
        if time.time() > deadline:
            break
        random.seed(seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        family = greedy_with_ordering(indices)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Strategy 2: Local search from best greedy solution
    if best_family and time.time() < deadline - 5:
        improved_family = local_search(best_family[:], max_iterations=20000)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family
    
    # Strategy 3: More aggressive local search with multiple restarts
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(1000 + seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        family = greedy_with_ordering(indices)
        
        # Local search
        improved_family = local_search(family, max_iterations=5000)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family
    
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
