import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Precompute comparability bitsets
    # Comp[a] is a bitset where bit b is 1 if a and b are comparable and a != b
    Comp = [0] * total_masks
    for a in range(total_masks):
        mask_a = 0
        for b in range(total_masks):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                mask_a |= (1 << b)
        Comp[a] = mask_a

    def calc_score(family, F_bits):
        # Sum of comparable pairs
        # Each pair is counted twice in the sum of degrees
        total_deg = 0
        for mask in family:
            total_deg += (Comp[mask] & F_bits).bit_count()
        return total_deg // 2

    def greedy_init():
        family = []
        F_bits = 0
        in_family = [False] * total_masks
        # Start with 0 (empty set) as it is subset of everything
        family.append(0)
        F_bits |= (1 << 0)
        in_family[0] = True
        
        for _ in range(m - 1):
            best_mask = -1
            best_count = -1
            # Randomize tie-breaking slightly to avoid determinism issues if needed, 
            # but greedy max is usually good.
            for mask in all_masks:
                if in_family[mask]:
                    continue
                count = (Comp[mask] & F_bits).bit_count()
                if count > best_count:
                    best_count = count
                    best_mask = mask
            family.append(best_mask)
            F_bits |= (1 << best_mask)
            in_family[best_mask] = True
        return family, F_bits

    def local_search_polish(family, F_bits, deadline):
        in_family = [False] * total_masks
        for mask in family:
            in_family[mask] = True
            
        current_score = calc_score(family, F_bits)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                if time.time() >= deadline:
                    break
                s_i = family[i]
                # Remove s_i from consideration for comparison
                F_bits_temp = F_bits & ~(1 << s_i)
                deg_i = (Comp[s_i] & F_bits_temp).bit_count()
                
                best_deg = -1
                best_mask = -1
                
                for mask in all_masks:
                    if in_family[mask]:
                        continue
                    deg_j = (Comp[mask] & F_bits_temp).bit_count()
                    if deg_j > best_deg:
                        best_deg = deg_j
                        best_mask = mask

                if best_mask != -1 and best_deg > deg_i:
                    # Swap
                    in_family[s_i] = False
                    in_family[best_mask] = True
                    F_bits = F_bits_temp | (1 << best_mask)
                    family[i] = best_mask
                    current_score += (best_deg - deg_i) // 2
                    improved = True
        
        return family, F_bits, current_score

    def simulated_annealing(family, F_bits, deadline):
        in_family = [False] * total_masks
        for mask in family:
            in_family[mask] = True
        
        available = [mask for mask in all_masks if not in_family[mask]]
        current_score = calc_score(family, F_bits)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        start_time = time.time()
        
        while time.time() < deadline:
            # Pick random index
            i = random.randint(0, m - 1)
            s_i = family[i]
            
            # Pick random mask from available
            if not available:
                break
            idx = random.randint(0, len(available) - 1)
            s_j = available[idx]
            
            # Calculate delta
            F_bits_temp = F_bits & ~(1 << s_i)
            deg_i = (Comp[s_i] & F_bits_temp).bit_count()
            deg_j = (Comp[s_j] & F_bits_temp).bit_count()
            
            delta = deg_j - deg_i
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                # Accept
                # Update family and structures
                in_family[s_i] = False
                in_family[s_j] = True
                
                # Update available list: remove s_j, add s_i
                # To keep it efficient, we can just swap in the list if we track indices, 
                # but simple remove/append is O(N) for list. 
                # N=4056. This might be slow if done millions of times.
                # Optimization: Use a set for in_family, and just pick random from a precomputed list of NOT in_family?
                # That requires rebuilding the list or checking.
                # Better: Keep a list of available masks. 
                # Removing an arbitrary element from a list is O(N).
                # We can swap s_j with the last element and pop.
                last_idx = len(available) - 1
                if idx != last_idx:
                    available[idx] = available[last_idx]
                available.pop()
                available.append(s_i)
                
                F_bits = F_bits_temp | (1 << s_j)
                family[i] = s_j
                current_score += delta // 2
                
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return family, F_bits, current_score

    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    # 1. Greedy Initialization
    family, F_bits = greedy_init()
    
    # 2. Simulated Annealing
    family, F_bits, score = simulated_annealing(family, F_bits, deadline - 5.0) # Leave 5s for polish
    
    # 3. Final Polish with Local Search
    family, F_bits, score = local_search_polish(family, F_bits, deadline)
    
    # Convert to lists of ints
    result = []
    for mask in family:
        result.append([i for i in range(n) if (mask >> i) & 1])
        
    return result
