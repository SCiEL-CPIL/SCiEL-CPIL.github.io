import time
import random
import copy

# Precomputation
N = 256
four_aps = []
three_aps = []
element_to_4aps = {i: [] for i in range(1, N+1)}
element_to_3aps = {i: [] for i in range(1, N+1)}

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        ap = (a, b, c, e)
        four_aps.append(ap)
        for elem in ap:
            element_to_4aps[elem].append(ap)

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        ap = (a, b, c)
        three_aps.append(ap)
        for elem in ap:
            element_to_3aps[elem].append(ap)

# Bitmask-based data structures
four_ap_conflict_masks = {i: [] for i in range(1, N+1)}
for ap in four_aps:
    a, b, c, e = ap
    for elem, others in [(a, (b, c, e)), (b, (a, c, e)), (c, (a, b, e)), (e, (a, b, c))]:
        mask = 0
        for o in others:
            mask |= (1 << (o - 1))
        four_ap_conflict_masks[elem].append(mask)

three_ap_partner_masks = {i: [] for i in range(1, N+1)}
for ap in three_aps:
    a, b, c = ap
    for elem, others in [(a, (b, c)), (b, (a, c)), (c, (a, b))]:
        mask = 0
        for o in others:
            mask |= (1 << (o - 1))
        three_ap_partner_masks[elem].append(mask)

def set_to_mask(A_set):
    mask = 0
    for x in A_set:
        mask |= (1 << (x - 1))
    return mask

def mask_to_set(mask):
    s = set()
    for i in range(1, N+1):
        if mask & (1 << (i-1)):
            s.add(i)
    return s

def is_4ap_free_addition_mask(A_mask, x):
    for cmask in four_ap_conflict_masks[x]:
        if (A_mask & cmask) == cmask:
            return False
    return True

def delta_3aps_for_addition_mask(A_mask, x):
    count = 0
    for pmask in three_ap_partner_masks[x]:
        if (A_mask & pmask) == pmask:
            count += 1
    return count

def count_3aps_mask(A_mask):
    count = 0
    for a, b, c in three_aps:
        if (A_mask & (1 << (a-1))) and (A_mask & (1 << (b-1))) and (A_mask & (1 << (c-1))):
            count += 1
    return count

def greedy_initial_mask(seed_val=0):
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        random.shuffle(candidates)
    for x in candidates:
        if is_4ap_free_addition_mask(A_mask, x):
            A_mask |= (1 << (x - 1))
    return A_mask

def local_search_mask(A_mask_init, deadline):
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Additions
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                continue
            if is_4ap_free_addition_mask(A_mask, x):
                delta = delta_3aps_for_addition_mask(A_mask, x)
                if delta > best_add_delta:
                    best_add_delta = delta
                    best_add_x = x
        
        if best_add_x != -1:
            A_mask |= (1 << (best_add_x - 1))
            current_score += best_add_delta
            improved = True
            continue
        
        # 2. Try Swaps (Optimized)
        # Precompute L_x for all x in A
        L_values = {}
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                L_values[x] = delta_3aps_for_addition_mask(A_mask & ~(1 << (x - 1)), x)
        
        best_swap = None
        best_swap_gain = 0
        
        # For each y not in A, check if it can be added to A\{x} for some x in A
        for y in range(1, N + 1):
            if A_mask & (1 << (y - 1)):
                continue
            
            # Calculate G_y: gain if y were in A
            G_y = delta_3aps_for_addition_mask(A_mask, y)
            
            if G_y <= 0:
                continue
                
            # Check 4-AP conflicts for y with respect to A
            # bad_aps: list of (other_mask, other_set) for 4-APs containing y where others are in A
            # If no bad_aps, y can be added to A directly (but we tried additions, so this implies G_y=0 or conflict)
            # Actually, if y can be added to A, it would have been caught in step 1.
            # So we assume y CANNOT be added to A.
            # This means there exists at least one 4-AP P containing y such that P\{y} subset A.
            
            critical_x_mask = 0
            has_conflict = False
            
            for cmask in four_ap_conflict_masks[y]:
                if (A_mask & cmask) == cmask:
                    has_conflict = True
                    # The elements in cmask are the ones that must be removed to make y addable.
                    # We can only remove one element x.
                    # So x must be one of the elements in cmask.
                    critical_x_mask |= cmask
            
            if not has_conflict:
                # This case should not happen if step 1 was exhaustive, but just in case
                # If no conflict, y can be added.
                # But step 1 would have found it.
                continue

            # We need to find x in A such that x is in critical_x_mask
            # And L_x < G_y
            
            # Iterate through bits in critical_x_mask that are also in A_mask
            candidates_x_mask = critical_x_mask & A_mask
            if candidates_x_mask == 0:
                continue
                
            # Find x in candidates_x_mask with minimum L_x
            min_L = float('inf')
            best_x = -1
            temp_mask = candidates_x_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                x = lsb.bit_length()
                if x in L_values:
                    if L_values[x] < min_L:
                        min_L = L_values[x]
                        best_x = x
                temp_mask ^= lsb
            
            if best_x != -1 and G_y - min_L > best_swap_gain:
                best_swap_gain = G_y - min_L
                best_swap = (best_x, y)
        
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~(1 << (x - 1))) | (1 << (y - 1))
            current_score += best_swap_gain
            improved = True
            continue
        
        # 3. Try Removals (only if score doesn't drop? No, removal drops score. 
        #    But maybe it enables a future swap? 
        #    Standard local search usually doesn't do pure removals unless it's a 2-opt style or simulated annealing.
        #    However, the original code had a polish pass with removals. 
        #    Let's stick to Additions and Swaps for the main loop to keep it simple and fast.
        #    If stuck, we might need to break out.
        
        if not improved:
            break
            
    return A_mask

def polish_pass(A_mask, deadline):
    current_score = count_3aps_mask(A_mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try removals
        # Removing x reduces score by L_x. 
        # It only helps if it allows a better addition/swap later.
        # This is a "hill climbing with restarts" or "simulated annealing" feature.
        # For now, let's just try to add back elements that were removed or find better swaps.
        
        # Try additions
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                continue
            if is_4ap_free_addition_mask(A_mask, x):
                delta = delta_3aps_for_addition_mask(A_mask, x)
                if delta > best_add_delta:
                    best_add_delta = delta
                    best_add_x = x
        
        if best_add_x != -1:
            A_mask |= (1 << (best_add_x - 1))
            current_score += best_add_delta
            improved = True
            continue
        
        # Try swaps (same logic as local_search)
        L_values = {}
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                L_values[x] = delta_3aps_for_addition_mask(A_mask & ~(1 << (x - 1)), x)
        
        best_swap = None
        best_swap_gain = 0
        
        for y in range(1, N + 1):
            if A_mask & (1 << (y - 1)):
                continue
            
            G_y = delta_3aps_for_addition_mask(A_mask, y)
            
            if G_y <= 0:
                continue
                
            critical_x_mask = 0
            has_conflict = False
            
            for cmask in four_ap_conflict_masks[y]:
                if (A_mask & cmask) == cmask:
                    has_conflict = True
                    critical_x_mask |= cmask
            
            if not has_conflict:
                continue

            candidates_x_mask = critical_x_mask & A_mask
            if candidates_x_mask == 0:
                continue
                
            min_L = float('inf')
            best_x = -1
            temp_mask = candidates_x_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                x = lsb.bit_length()
                if x in L_values:
                    if L_values[x] < min_L:
                        min_L = L_values[x]
                        best_x = x
                temp_mask ^= lsb
            
            if best_x != -1 and G_y - min_L > best_swap_gain:
                best_swap_gain = G_y - min_L
                best_swap = (best_x, y)
        
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~(1 << (x - 1))) | (1 << (y - 1))
            current_score += best_swap_gain
            improved = True
            continue
        
        break
        
    return A_mask

def crossover_mask(parent1, parent2, rng):
    """Uniform crossover: for each position, pick from parent1 or parent2."""
    child = 0
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        if rng.random() < 0.5:
            if parent1 & bit:
                child |= bit
        else:
            if parent2 & bit:
                child |= bit
    return child

def repair_4ap_free(mask, rng):
    """Remove elements that create 4-APs, keeping as many as possible."""
    result = mask
    for x in range(1, N + 1):
        if not (result & (1 << (x - 1))):
            continue
        # Check if x creates a 4-AP with other elements in result
        for cmask in four_ap_conflict_masks[x]:
            if (result & cmask) == cmask:
                # Remove x (or one of the conflicting elements)
                # For simplicity, remove x
                result &= ~(1 << (x - 1))
                break
    return result

def mutate_mask(mask, rng, mutation_rate=0.1):
    """Apply random mutations: add, remove, or swap elements."""
    result = mask
    num_mutations = max(1, int(N * mutation_rate))
    
    for _ in range(num_mutations):
        op = rng.random()
        if op < 0.4:
            # Add a random element
            x = rng.randint(1, N)
            if not (result & (1 << (x - 1))):
                if is_4ap_free_addition_mask(result, x):
                    result |= (1 << (x - 1))
        elif op < 0.7:
            # Remove a random element
            x = rng.randint(1, N)
            if result & (1 << (x - 1)):
                result &= ~(1 << (x - 1))
        else:
            # Swap: remove one, add another
            x = rng.randint(1, N)
            if result & (1 << (x - 1)):
                temp = result & ~(1 << (x - 1))
                y = rng.randint(1, N)
                if y != x and not (temp & (1 << (y - 1))):
                    if is_4ap_free_addition_mask(temp, y):
                        result = temp | (1 << (y - 1))
    
    return result

def genetic_algorithm(initial_population, deadline, pop_size=12, elite_size=3):
    """
    Genetic algorithm to evolve 4-AP-free sets with high 3-AP count.
    """
    rng = random.Random(42)
    
    # Initialize population with scores
    population = []
    for mask in initial_population:
        score = count_3aps_mask(mask)
        population.append((mask, score))
    
    # Pad population if needed
    while len(population) < pop_size:
        # Create new individuals by mutating existing ones
        parent = rng.choice(population)
        child = mutate_mask(parent[0], rng, 0.2)
        child = repair_4ap_free(child, rng)
        score = count_3aps_mask(child)
        population.append((child, score))
    
    # Sort by score descending
    population.sort(key=lambda x: x[1], reverse=True)
    
    best_mask, best_score = population[0]
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Elitism: keep top individuals
        new_population = [population[i] for i in range(min(elite_size, len(population)))]
        
        # Create offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = rng.sample(population, min(3, len(population)))
                return max(candidates, key=lambda x: x[1])[0]
            
            p1 = select()
            p2 = select()
            
            # Crossover
            if rng.random() < 0.7:
                child = crossover_mask(p1, p2, rng)
            else:
                child = p1
            
            # Mutation
            if rng.random() < 0.8:
                child = mutate_mask(child, rng, 0.15)
            
            # Repair
            child = repair_4ap_free(child, rng)
            
            score = count_3aps_mask(child)
            new_population.append((child, score))
        
        # Sort and update
        new_population.sort(key=lambda x: x[1], reverse=True)
        population = new_population
        
        if population[0][1] > best_score:
            best_score = population[0][1]
            best_mask = population[0][0]
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s budget, 5s safety margin
    
    best_mask = None
    best_score = -1
    
    # Generate diverse initial candidates using bitmask greedy
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial_mask(0))
    
    # Candidates 1-10: Randomized Greedies
    for i in range(1, 11):
        if time.time() > deadline - 20:
            break
        candidates.append(greedy_initial_mask(i))
    
    # Run local search on a few candidates first to get good starting points
    optimized_candidates = []
    for idx, init_mask in enumerate(candidates[:5]):
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time()
        if remaining <= 5:
            break
        search_deadline = min(deadline - 20, time.time() + max(2, remaining / 6))
        optimized = local_search_mask(init_mask, search_deadline)
        optimized_candidates.append(optimized)
    
    # Add optimized candidates to the pool
    all_candidates = candidates + optimized_candidates
    
    # Run genetic algorithm with a portion of the time budget
    ga_deadline = min(deadline - 15, time.time() + 30)
    if ga_deadline > time.time() + 5:
        ga_population = all_candidates[:min(12, len(all_candidates))]
        ga_result = genetic_algorithm(ga_population, ga_deadline, pop_size=12, elite_size=3)
        
        ga_score = count_3aps_mask(ga_result)
        if ga_score > best_score:
            best_score = ga_score
            best_mask = ga_result
    
    # Run local search on remaining candidates
    for idx, init_mask in enumerate(all_candidates):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining <= 3:
            break
        search_deadline = min(deadline - 5, time.time() + max(1, remaining / 4))
        optimized = local_search_mask(init_mask, search_deadline)
        opt_score = count_3aps_mask(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = optimized
    
    # Final polish on the best set
    if best_mask is not None:
        final_deadline = deadline - 2
        best_mask = polish_pass(best_mask, final_deadline)
    
    if best_mask is None:
        best_mask = candidates[0] if candidates else greedy_initial_mask(0)
    
    # Convert to sorted list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
    
    return sorted(result)
