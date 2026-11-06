import time
import random
import numpy as np

N = 512

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but keep margin)
    
    # 1. Precompute 4-AP constraints
    # For each index x (0..N-1), we want to know which other indices y, z, w 
    # form a 4-AP with x.
    # A 4-AP is a, a+d, a+2d, a+3d.
    # If we have a set S, adding x creates a 4-AP if there exist y, z, w in S 
    # such that {x, y, z, w} is a 4-AP.
    
    # Let's precompute for each x, the list of "forbidden triples" (y, z, w).
    # If y, z, w are in S, x cannot be added.
    
    forbidden = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # 0-indexed
            i, j, k, l = a-1, b-1, c-1, e-1
            
            forbidden[i].append((j, k, l))
            forbidden[j].append((i, k, l))
            forbidden[k].append((i, j, l))
            forbidden[l].append((i, j, k))
            
    # Convert to lists of masks for faster intersection
    forbidden_masks = []
    for i in range(N):
        masks = []
        for t in forbidden[i]:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(m)
        forbidden_masks.append(masks)

    def can_add(bits, x):
        # Check if any forbidden mask for x is a subset of bits
        for m in forbidden_masks[x]:
            if (bits & m) == m:
                return False
        return True

    def repair(bits):
        """
        Repair an invalid bitset by removing elements from 4-APs.
        Returns a valid bitset.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find any 4-AP in the set
            # We can iterate through all possible 4-APs and check if they are in bits
            # To speed up, we can iterate through elements and check if they are part of a 4-AP
            # But since we need to remove one element from each 4-AP, let's just find one 4-AP.
            
            # Efficient way: iterate through all 4-APs (a, b, c, d) and check if all are in bits.
            # If so, remove one of them (e.g., the last one) and set changed = True.
            # This might be slow if there are many 4-APs, but repair is called on children
            # which are only slightly invalid (due to crossover).
            
            # Optimization: Only check 4-APs involving elements that are in bits?
            # Or just scan all 4-APs. Number of 4-APs is ~ N^2 / 6 ~ 40k. 
            # Checking each is 4 bit ops. 40k * 4 = 160k ops. Very fast.
            
            for d in range(1, (N - 1) // 3 + 1):
                for a in range(1, N - 3 * d + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    
                    i, j, k, l = a-1, b-1, c-1, e-1
                    
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    if (bits & mask) == mask:
                        # Found a 4-AP. Remove one element.
                        # Heuristic: remove the element that is part of the most 4-APs?
                        # Or just remove a random one?
                        # Let's remove the last one for simplicity, or maybe the one with highest index?
                        # Let's remove a random one to introduce variation.
                        # Actually, removing any one breaks this specific 4-AP.
                        # Let's remove 'l' (the last one) for determinism in this step, 
                        # or maybe 'i' (the first one).
                        # Let's try removing the one that appears in the most other potential 4-APs?
                        # Too expensive. Let's just remove 'l'.
                        bits &= ~(1 << l)
                        changed = True
                        break
                if changed:
                    break
        return bits

    def greedy_add(bits):
        size = bin(bits).count('1')
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order to avoid bias
            candidates = [a for a in range(N) if not (bits & (1 << a))]
            random.shuffle(candidates)
            for a in candidates:
                if time.time() > deadline:
                    break
                if not (bits & (1 << a)):
                    if can_add(bits, a):
                        bits |= (1 << a)
                        size += 1
                        changed = True
        return bits, size

    def local_search(bits, max_iters=200):
        """
        Perform local search (swaps) to improve the set.
        """
        bits, current_size = greedy_add(bits)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            candidates = [a for a in range(N) if not (bits & (1 << a))]
            random.shuffle(candidates)
            
            for a in candidates:
                if time.time() > deadline:
                    break
                    
                if bits & (1 << a):
                    continue
                
                # Find bad triples for a
                bad_triples = []
                for t in forbidden[a]:
                    m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                    if (bits & m) == m:
                        bad_triples.append(t)
                
                if not bad_triples:
                    # Can add directly
                    bits |= (1 << a)
                    current_size += 1
                    improved = True
                    # Restart greedy add
                    bits, current_size = greedy_add(bits)
                    break
                
                # Find intersection of indices in bad_triples
                common = set(bad_triples[0])
                for t in bad_triples[1:]:
                    common &= set(t)
                
                if common:
                    # We can remove any r in common
                    r = random.choice(list(common))
                    
                    # Perform swap
                    bits &= ~(1 << r)
                    bits |= (1 << a)
                    # Size remains same
                    improved = True
                    # Restart greedy add
                    bits, current_size = greedy_add(bits)
                    break
            
            if not improved:
                # Try a kick: remove 1 random element, try to add 1
                # Or just break
                break
                
        return bits, current_size

    # 2. Genetic Algorithm Setup
    
    # Initialize population
    pop_size = 20
    population = []
    
    # First individual: Ternary Digit Heuristic
    init_set = set()
    for num in range(1, N + 1):
        x = num
        valid = True
        while x > 0:
            if x % 3 == 0:
                valid = False
                break
            x //= 3
        if valid:
            init_set.add(num - 1)
            
    bits = 0
    for i in init_set:
        bits |= (1 << i)
    bits, size = greedy_add(bits)
    population.append((bits, size))
    
    # Other individuals: Random Greedy
    for _ in range(pop_size - 1):
        if time.time() > deadline:
            break
        b = 0
        # Random greedy: pick random elements, add if valid
        # This might be slow if done naively.
        # Let's just start with an empty set and do greedy_add with random order?
        # greedy_add already randomizes.
        # But greedy_add from empty set is the same as the first run if random seed is same?
        # No, random.shuffle uses global state.
        # Let's just run greedy_add from empty set.
        b, s = greedy_add(0)
        population.append((b, s))
        
    # Sort population by size (descending)
    population.sort(key=lambda x: x[1], reverse=True)
    
    best_bits = population[0][0]
    best_size = population[0][1]
    
    # 3. Evolution Loop
    
    num_generations = 50 # Max, but limited by time
    
    for gen in range(num_generations):
        if time.time() > deadline:
            break
            
        new_population = []
        
        # Elitism: Keep the best 2
        for i in range(2):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Selection: Tournament selection
            def select():
                idx1 = random.randint(0, len(population) - 1)
                idx2 = random.randint(0, len(population) - 1)
                if population[idx1][1] >= population[idx2][1]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: Uniform crossover with block repair?
            # Simple uniform crossover: for each bit, pick from parent1 or parent2 randomly.
            # This will likely create invalid sets.
            
            child_bits = 0
            for i in range(N):
                if random.random() < 0.5:
                    if parent1[0] & (1 << i):
                        child_bits |= (1 << i)
                else:
                    if parent2[0] & (1 << i):
                        child_bits |= (1 << i)
            
            # Repair: Remove elements from 4-APs
            child_bits = repair(child_bits)
            
            # Local Search
            child_bits, child_size = local_search(child_bits, max_iters=50)
            
            new_population.append((child_bits, child_size))
            
            if child_size > best_size:
                best_size = child_size
                best_bits = child_bits
        
        # Update population
        # Sort new population
        new_population.sort(key=lambda x: x[1], reverse=True)
        
        # Truncate to pop_size
        population = new_population[:pop_size]
        
        # Update best
        if population[0][1] > best_size:
            best_size = population[0][1]
            best_bits = population[0][0]

    # Final conversion
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result

# EVOLVE-BLOCK-END
