import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Generate all k-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total_sets = len(all_sets)
    
    # Precompute pairwise intersections for all pairs
    # This is C(1820, 2) ≈ 1.66M pairs, each intersection is just a bitwise AND
    # We'll store in a dict or 2D array for fast lookup
    # Actually, let's precompute intersection[a][b] for all pairs
    # Using a dictionary keyed by (min_idx, max_idx) to save memory
    # Or just compute on the fly since it's fast with bitmasks
    
    def can_add_fast(F, C):
        """Check if adding C to F creates a sunflower."""
        m = len(F)
        for i in range(m):
            a = F[i]
            inter_a_c = a & C
            for j in range(i + 1, m):
                b = F[j]
                inter_a_b = a & b
                if inter_a_b == inter_a_c:
                    # Check if inter_b_c also equals
                    if (b & C) == inter_a_b:
                        return False
        return True
    
    def greedy_with_order(order, family):
        """Try to extend family with sets in given order."""
        for idx in order:
            C = all_sets[idx]
            if can_add_fast(family, C):
                family.append(C)
        return family
    
    best_family = []
    best_size = 0
    rng = random.Random(12345)
    
    # Phase 1: Multiple random greedy restarts
    while time.time() < deadline - 20.0:
        shuffled_indices = list(range(total_sets))
        rng.shuffle(shuffled_indices)
        
        current_family = []
        for idx in shuffled_indices:
            C = all_sets[idx]
            if can_add_fast(current_family, C):
                current_family.append(C)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
    
    # Phase 2: Local search / simulated annealing on the best family
    if time.time() < deadline - 5.0 and best_size > 0:
        current_family = best_family[:]
        current_size = best_size
        
        # Build a set of masks in current family for O(1) lookup
        family_set = set(current_family)
        
        # Precompute which sets are NOT in the family
        available_sets = [s for s in all_sets if s not in family_set]
        
        # Simulated annealing: try to swap out one set and swap in another
        temperature = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        while time.time() < deadline - 3.0 and temperature > min_temp:
            progress = False
            
            # Randomly pick a set to remove
            if current_family:
                rem_idx = rng.randint(0, len(current_family) - 1)
                rem_set = current_family[rem_idx]
                
                # Try to find a set to add
                if available_sets:
                    # Try random available sets
                    attempts = min(50, len(available_sets))
                    for _ in range(attempts):
                        add_idx = rng.randint(0, len(available_sets) - 1)
                        add_set = available_sets[add_idx]
                        
                        if add_set == rem_set:
                            continue
                        
                        # Check if removing rem_set and adding add_set is valid
                        # First, check if add_set conflicts with current_family minus rem_set
                        new_family = [s for i, s in enumerate(current_family) if i != rem_idx]
                        
                        if can_add_fast(new_family, add_set):
                            # Valid move: size stays the same
                            current_family = new_family + [add_set]
                            family_set.discard(rem_set)
                            family_set.add(add_set)
                            # Update available sets
                            available_sets[add_idx] = rem_set
                            progress = True
                            break
            
            if not progress:
                # Try to add a set without removing (increases size)
                if available_sets:
                    attempts = min(20, len(available_sets))
                    for _ in range(attempts):
                        add_idx = rng.randint(0, len(available_sets) - 1)
                        add_set = available_sets[add_idx]
                        
                        if can_add_fast(current_family, add_set):
                            current_family.append(add_set)
                            family_set.add(add_set)
                            available_sets[add_idx] = -1  # Mark as used (not in available)
                            # Actually we need to remove it from available properly
                            # Let's rebuild available sets periodically
                            current_size = len(current_family)
                            if current_size > best_size:
                                best_size = current_size
                                best_family = current_family[:]
                            progress = True
                            break
            
            temperature *= cooling_rate
            
            # Rebuild available sets periodically to keep them clean
            if int(temperature * 1000) % 100 == 0 or temperature < min_temp * 2:
                family_set = set(current_family)
                available_sets = [s for s in all_sets if s not in family_set]
    
    # Phase 3: Final greedy polish - try to add any remaining sets
    if time.time() < deadline - 1.0:
        family_set = set(best_family)
        remaining = [s for s in all_sets if s not in family_set]
        for s in remaining:
            if time.time() > deadline - 1.0:
                break
            if can_add_fast(best_family, s):
                best_family.append(s)
    
    return [sorted([i for i in range(n) if (mask >> i) & 1]) for mask in best_family]
