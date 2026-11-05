import time
import random
import math
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 52.0  # Safety margin before 90s kill

    # 1. Precompute all 3-APs as (a, b, c) tuples where a < b < c and b-a = c-b
    three_aps = []
    for a in range(N):
        max_d = (N - 1 - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
    three_aps = np.array(three_aps, dtype=np.int32)  # shape (M, 3)
    M = len(three_aps)

    # 2. Precompute 4-AP conflicts: for each x, list of (y, z, w) such that {x,y,z,w} is a 4-AP
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    # 3. Precompute for each element x, the list of 3-AP indices where x participates
    #    and its role (0=start, 1=middle, 2=end)
    x_ap_indices = [[] for _ in range(N)]
    for idx, (a, b, c) in enumerate(three_aps):
        x_ap_indices[a].append(idx)
        x_ap_indices[b].append(idx)
        x_ap_indices[c].append(idx)

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps_numpy(mask):
        """Count 3-APs using numpy vectorized operations."""
        if M == 0:
            return 0
        # Convert mask to array of element indicators
        # For each 3-AP (a,b,c), check if all three bits are set
        a = three_aps[:, 0]
        b = three_aps[:, 1]
        c = three_aps[:, 2]
        # Check bits
        a_set = (mask >> a) & 1
        b_set = (mask >> b) & 1
        c_set = (mask >> c) & 1
        return int(np.sum(a_set & b_set & c_set))

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    # 4. Greedy initialization with 3-AP bias
    def greedy_init():
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                d = delta_3aps_add(mask, x)
                # Accept with probability based on delta
                if d > 0:
                    mask |= (1 << x)
                elif random.random() < 0.4:
                    mask |= (1 << x)
        return mask

    # 5. Repair function: given a mask that may have 4-APs, remove elements to make it valid
    def repair(mask):
        """Remove elements to eliminate all 4-APs. Greedy removal of most-conflicting elements."""
        if mask == 0:
            return mask
        
        # Find all 4-APs in the mask
        # For each 4-AP, we need to remove at least one element
        # Strategy: repeatedly find a 4-AP, remove the element that participates in most 4-APs
        changed = True
        while changed:
            changed = False
            # Find all 4-APs present in mask
            four_aps = []
            for a in range(N):
                if not ((mask >> a) & 1):
                    continue
                max_d = (N - 1 - a) // 3
                for d in range(1, max_d + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> e) & 1):
                        four_aps.append((a, b, c, e))
            
            if not four_aps:
                break
            
            # Count how many 4-APs each element participates in
            conflict_count = {}
            for (a, b, c, e) in four_aps:
                for x in (a, b, c, e):
                    conflict_count[x] = conflict_count.get(x, 0) + 1
            
            # Remove the element with highest conflict count
            worst = max(conflict_count, key=conflict_count.get)
            mask &= ~(1 << worst)
            changed = True
        
        return mask

    # 6. Crossover: take union of two parents, then repair
    def crossover(parent1, parent2):
        child = parent1 | parent2
        child = repair(child)
        return child

    # 7. Mutation: add/remove/swap elements
    def mutate(mask, rate=0.3):
        new_mask = mask
        num_mutations = int(random.random() * 3) + 1  # 1-3 mutations
        
        for _ in range(num_mutations):
            op = random.random()
            if op < 0.4:
                # Add
                x = random.randint(0, N - 1)
                if not ((new_mask >> x) & 1) and is_valid_add(new_mask, x):
                    new_mask |= (1 << x)
            elif op < 0.7:
                # Remove
                x = random.randint(0, N - 1)
                if (new_mask >> x) & 1:
                    new_mask &= ~(1 << x)
            else:
                # Swap
                x = random.randint(0, N - 1)
                if (new_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((new_mask >> y) & 1):
                        temp = new_mask & ~(1 << x)
                        if is_valid_add(temp, y):
                            new_mask = temp | (1 << y)
        
        return new_mask

    # 8. Targeted mutation: bias toward high-delta additions
    def targeted_mutate(mask):
        new_mask = mask
        # Try adding a few elements with high delta
        candidates = []
        for x in range(N):
            if not ((mask >> x) & 1) and is_valid_add(mask, x):
                d = delta_3aps_add(mask, x)
                if d > 0:
                    candidates.append((d, x))
        
        # Sort by delta descending, take top few
        candidates.sort(reverse=True)
        num_adds = min(len(candidates), random.randint(1, 3))
        for i in range(num_adds):
            d, x = candidates[i]
            if is_valid_add(new_mask, x):
                new_mask |= (1 << x)
        
        # Also try removing low-contribution elements
        for x in range(N):
            if (new_mask >> x) & 1:
                d = delta_3aps_add(new_mask, x)  # This is the contribution
                if d == 0 and random.random() < 0.3:
                    new_mask &= ~(1 << x)
        
        return new_mask

    # 9. Genetic Algorithm
    pop_size = 40
    population = []
    scores = []
    
    # Initialize population
    for i in range(pop_size):
        m = greedy_init()
        s = count_3aps_numpy(m)
        population.append(m)
        scores.append(s)
    
    best_mask = max(population, key=lambda m: scores[population.index(m)])
    best_score = max(scores)

    elite_count = 5
    generation = 0
    
    while time.time() < deadline:
        # Sort by fitness
        idx_sorted = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        population = [population[i] for i in idx_sorted]
        scores = [scores[i] for i in idx_sorted]
        
        # Keep elites
        new_population = population[:elite_count].copy()
        new_scores = scores[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 1.0:
                break
            
            # Selection: tournament
            def tournament_select():
                tournament = random.sample(range(len(population)), min(3, len(population)))
                winner = max(tournament, key=lambda i: scores[i])
                return population[winner]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover or mutation
            if random.random() < 0.5:
                child = crossover(p1, p2)
            else:
                child = p1 | p2  # Union
                child = repair(child)
            
            # Mutation
            if random.random() < 0.7:
                child = mutate(child)
            elif random.random() < 0.5:
                child = targeted_mutate(child)
            
            # Ensure non-empty
            if child == 0:
                child = 1  # At least element 0
            
            s = count_3aps_numpy(child)
            new_population.append(child)
            new_scores.append(s)
            
            if s > best_score:
                best_score = s
                best_mask = child
        
        population = new_population
        scores = new_scores
        generation += 1
        
        # Periodic diversity injection
        if generation % 20 == 0 and time.time() < deadline - 5.0:
            fresh = greedy_init()
            fs = count_3aps_numpy(fresh)
            # Replace worst individual
            worst_idx = scores.index(min(scores))
            population[worst_idx] = fresh
            scores[worst_idx] = fs
            if fs > best_score:
                best_score = fs
                best_mask = fresh

    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
