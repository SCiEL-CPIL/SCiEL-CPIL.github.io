import time
import random

N = 256

# ---- Precompute per-element 3-AP pair lists and 4-AP conflict triples ----
f3 = [[] for _ in range(N + 1)]   # x as FIRST term of a 3-AP: (x+d, x+2d)
m3 = [[] for _ in range(N + 1)]   # x as MIDDLE term: (x-d, x+d)
l3 = [[] for _ in range(N + 1)]   # x as LAST term: (x-2d, x-d)
f4 = [[] for _ in range(N + 1)]   # 4-APs containing x: tuple of the OTHER three elements

for x in range(1, N + 1):
    # x first: x+2d <= N
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    # x middle: x-d >= 1 and x+d <= N
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    # x last: x-2d >= 1
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))
    # x in a 4-AP, four positions
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4[x].append((x + d, x + 2 * d, x + 3 * d))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4[x].append((x - d, x + d, x + 2 * d))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4[x].append((x - 2 * d, x - d, x + d))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4[x].append((x - 3 * d, x - 2 * d, x - d))

# ---- Precompute bitmask structures for fast GA evaluation ----
# conflict_mask[x]: bitmask of elements that with x form a 4-AP (the other 3 elements)
# For adding x to set S: valid iff (S & conflict_mask[x]) has < 3 bits set
conflict_mask = [0] * (N + 1)
for x in range(1, N + 1):
    mask = 0
    for (a, b, c) in f4[x]:
        mask |= (1 << a) | (1 << b) | (1 << c)
    conflict_mask[x] = mask

# Precompute 3-AP contribution masks for each element
# For element x, its 3-APs are formed with pairs from f3[x], m3[x], l3[x]
# We precompute for each x a list of (pair_mask, count) where pair_mask identifies the pair
# Actually, let's precompute: for each x, list of bitmasks representing each 3-AP containing x
# where the mask has exactly 2 bits (the other two elements)
ap3_masks = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    seen = set()
    for (a, b) in f3[x]:
        m = (1 << a) | (1 << b)
        if m not in seen:
            seen.add(m)
            ap3_masks[x].append(m)
    for (a, b) in m3[x]:
        m = (1 << a) | (1 << b)
        if m not in seen:
            seen.add(m)
            ap3_masks[x].append(m)
    for (a, b) in l3[x]:
        m = (1 << a) | (1 << b)
        if m not in seen:
            seen.add(m)
            ap3_masks[x].append(m)

def count_3AP_total(present):
    s = 0
    for x in range(1, N + 1):
        if present[x]:
            for (a, b) in m3[x]:
                if present[a] and present[b]:
                    s += 1
    return s

def bitmask_fitness(mask):
    """Count 3-APs in the set represented by bitmask. Returns (count, valid)."""
    count = 0
    # Count each 3-AP exactly once by counting middle terms
    # For each x in set, count pairs (x-d, x+d) both in set
    m = mask
    while m:
        low_bit = m & (-m)
        x = low_bit.bit_length()  # 1-indexed position
        # Count 3-APs where x is the middle
        for (a, b) in m3[x]:
            if (mask & (1 << a)) and (mask & (1 << b)):
                count += 1
        m ^= low_bit
    return count

def is_valid_mask(mask):
    """Check if bitmask represents a 4-AP-free set."""
    m = mask
    while m:
        low_bit = m & (-m)
        x = low_bit.bit_length()
        # Check if x with any 3 others in set form a 4-AP
        # x is in 4-AP with (a,b,c) from f4[x] if all three are in set
        for (a, b, c) in f4[x]:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                return False
        m ^= low_bit
    return True

def mask_to_list(mask):
    """Convert bitmask to sorted list of elements."""
    result = []
    m = mask
    while m:
        low_bit = m & (-m)
        x = low_bit.bit_length()
        result.append(x)
        m ^= low_bit
    return result

def list_to_mask(lst):
    """Convert list to bitmask."""
    mask = 0
    for x in lst:
        mask |= (1 << x)
    return mask

def greedy_seed_mask(rng):
    """Generate a random 4-AP-free set as bitmask using greedy approach."""
    mask = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        # Check if adding x creates a 4-AP
        valid = True
        for (a, b, c) in f4[x]:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                valid = False
                break
        if valid:
            mask |= (1 << x)
    return mask

def local_search_mask(mask, deadline, rng):
    """Local search on bitmask representation with add/remove/swap moves."""
    current_score = bitmask_fitness(mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try all single additions
        best_add = None
        best_add_gain = 0
        for x in range(1, N + 1):
            if mask & (1 << x):
                continue
            # Check validity
            valid = True
            for (a, b, c) in f4[x]:
                if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                    valid = False
                    break
            if not valid:
                continue
            # Compute gain: 3-APs created by adding x
            gain = 0
            for (a, b) in m3[x]:
                if (mask & (1 << a)) and (mask & (1 << b)):
                    gain += 1
            if gain > best_add_gain:
                best_add_gain = gain
                best_add = x
        
        if best_add is not None:
            mask |= (1 << best_add)
            current_score += best_add_gain
            improved = True
        
        # Try all single removals + additions (swaps)
        if not improved:
            best_swap = None
            best_swap_gain = 0
            for r in range(1, N + 1):
                if not (mask & (1 << r)):
                    continue
                # Loss from removing r
                loss = 0
                for (a, b) in m3[r]:
                    if (mask & (1 << a)) and (mask & (1 << b)):
                        loss += 1
                # Try adding each x
                for x in range(1, N + 1):
                    if mask & (1 << x):
                        continue
                    # Check validity after removing r
                    new_mask = mask & ~(1 << r)
                    valid = True
                    for (a, b, c) in f4[x]:
                        if (new_mask & (1 << a)) and (new_mask & (1 << b)) and (new_mask & (1 << c)):
                            valid = False
                            break
                    if not valid:
                        continue
                    gain = 0
                    for (a, b) in m3[x]:
                        if (new_mask & (1 << a)) and (new_mask & (1 << b)):
                            gain += 1
                    net = gain - loss
                    if net > best_swap_gain:
                        best_swap_gain = net
                        best_swap = (r, x)
            
            if best_swap is not None and best_swap_gain > 0:
                r, x = best_swap
                mask = (mask & ~(1 << r)) | (1 << x)
                current_score += best_swap_gain
                improved = True
        
        if not improved:
            break
    
    return mask, current_score

def genetic_algorithm(deadline, rng):
    """Genetic algorithm with bitmask-based fitness."""
    pop_size = 20
    elite_size = 4
    mutation_rate = 0.15
    crossover_prob = 0.7
    
    # Initialize population
    population = []
    scores = []
    for _ in range(pop_size):
        m = greedy_seed_mask(rng)
        population.append(m)
        scores.append(bitmask_fitness(m))
    
    best_mask = population[scores.index(max(scores))]
    best_score = max(scores)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        sorted_idx = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        # Track best
        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = population[0]
        
        # Create new population
        new_pop = []
        new_scores = []
        
        # Elitism
        for i in range(min(elite_size, pop_size)):
            new_pop.append(population[i])
            new_scores.append(scores[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size and time.time() < deadline:
            # Select parents (tournament selection)
            p1_idx = rng.randint(0, pop_size - 1)
            p2_idx = rng.randint(0, pop_size - 1)
            # Tournament
            for _ in range(3):
                c1 = rng.randint(0, pop_size - 1)
                if scores[c1] > scores[p1_idx]:
                    p1_idx = c1
            for _ in range(3):
                c2 = rng.randint(0, pop_size - 1)
                if scores[c2] > scores[p2_idx]:
                    p2_idx = c2
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            if rng.random() < crossover_prob:
                # Uniform crossover
                child = 0
                for x in range(1, N + 1):
                    if rng.random() < 0.5:
                        child |= (p1 & (1 << x))
                    else:
                        child |= (p2 & (1 << x))
            else:
                child = p1
            
            # Mutation: random flips
            if rng.random() < mutation_rate:
                # Add random element
                for _ in range(rng.randint(1, 3)):
                    x = rng.randint(1, N)
                    if not (child & (1 << x)):
                        # Check validity
                        valid = True
                        for (a, b, c) in f4[x]:
                            if (child & (1 << a)) and (child & (1 << b)) and (child & (1 << c)):
                                valid = False
                                break
                        if valid:
                            child |= (1 << x)
                # Remove random element
                for _ in range(rng.randint(0, 2)):
                    m = child
                    if m:
                        # Pick random set bit
                        bits = []
                        tmp = child
                        while tmp:
                            low = tmp & (-tmp)
                            bits.append(low.bit_length())
                            tmp ^= low
                        if bits:
                            x = rng.choice(bits)
                            child &= ~(1 << x)
            
            # Repair: remove elements that create 4-APs
            changed = True
            while changed:
                changed = False
                for x in range(1, N + 1):
                    if child & (1 << x):
                        for (a, b, c) in f4[x]:
                            if (child & (1 << a)) and (child & (1 << b)) and (child & (1 << c)):
                                child &= ~(1 << x)
                                changed = True
                                break
            
            if child == 0:
                child = 1 << 1  # At least element 1
            
            sc = bitmask_fitness(child)
            new_pop.append(child)
            new_scores.append(sc)
        
        population = new_pop
        scores = new_scores
        
        # Periodic local search on best individual
        if generation % 5 == 0 and time.time() < deadline - 5:
            ls_mask, ls_score = local_search_mask(population[0], time.time() + 2.0, rng)
            if ls_score > scores[0]:
                population[0] = ls_mask
                scores[0] = ls_score
                if ls_score > best_score:
                    best_score = ls_score
                    best_mask = ls_mask
    
    return best_mask, best_score

def construct():
    start = time.time()
    # safety margin well under the 60s eval / 90s kill
    deadline = start + 52.0
    rng = random.Random(12345)

    best_mask = None
    best_score = -1

    # Run genetic algorithm
    ga_mask, ga_score = genetic_algorithm(deadline - 8.0, rng)
    if ga_score > best_score:
        best_score = ga_score
        best_mask = ga_mask

    # Local search polish on the GA result
    if time.time() < deadline - 2.0 and best_mask is not None:
        ls_mask, ls_score = local_search_mask(best_mask, deadline - 0.5, rng)
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask

    # Also try a few greedy seeds with local search
    for _ in range(3):
        if time.time() > deadline - 2.0:
            break
        gm = greedy_seed_mask(rng)
        ls_mask, ls_score = local_search_mask(gm, deadline - 0.5, rng)
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask

    if best_mask is None:
        best_mask = greedy_seed_mask(random.Random(0))

    A = mask_to_list(best_mask)
    if not A:
        A = [1]

    # hard validity guarantee: drop any element that sits in a 4-AP
    present = [False] * (N + 1)
    for x in A:
        present[x] = True
    for x in list(A):
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                present[x] = False
                A.remove(x)
                break
    if not A:
        A = [1]
    return A
