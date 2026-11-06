import time
import random
import numpy as np
from scipy.fft import fft, ifft

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    n_target = 32
    max_val = 512
    max_d = max_val // 2  # 255
    
    # 1. Precompute the "Progression Kernel" using FFT.
    # We want to know, for any potential new element x, how many new 3-APs it creates.
    # A 3-AP is (a, a+d, a+2d). 
    # If we add x, it can be:
    # - The start: x, x+d, x+2d (needs x+d, x+2d in S)
    # - The middle: x-d, x, x+d (needs x-d, x+d in S)
    # - The end: x-2d, x-d, x (needs x-2d, x-d in S)
    
    # Instead of checking S directly, we can precompute a global structure?
    # No, S changes. But we can speed up the evaluation of a specific S.
    
    # Let's use a bitset representation for S: S is a numpy array of 0s and 1s of length max_val.
    # Score(S) = sum_{d=1}^{255} (exists a: S[a] & S[a+d] & S[a+2d])
    
    # We can compute the number of 3-APs for a given S using FFT?
    # Let S be a vector.
    # The number of 3-APs is related to the autocorrelation?
    # Actually, the current bitset loop is already quite fast (255 iterations of bitwise ops on 512-bit ints).
    # ~10-20 microseconds per score.
    # 145s / 20us = ~7 million evaluations.
    # That is plenty for SA.
    
    # The "different strategy" requested is FFT.
    # Let's see if FFT helps.
    # If we define A(z) = sum S[i] z^i.
    # Then A(z)^2 has coefficients representing sums S[i]S[j].
    # This doesn't directly give 3-APs which are S[i]S[i+d]S[i+2d].
    # This is a "3-term correlation".
    # It can be computed as:
    # For each d, count = sum S[i] S[i+d] S[i+2d].
    # This is not a standard convolution.
    
    # However, we can use the FFT to speed up the *local search moves* if we batch them?
    # Or maybe the "structural change" is to use the FFT to find *good initial sets*?
    
    # Let's try a different structural change: **Genetic Algorithm with Bitset Crossover**.
    # The prompt mentions "operator 'cross'".
    # The current program uses random restarts and hill climbing.
    # A GA maintains a population and uses crossover/mutation.
    
    # Strategy:
    # 1. Initialize a population of 50 random sets.
    # 2. Evaluate all.
    # 3. Selection: Keep top 10.
    # 4. Crossover: Bitwise XOR or Uniform Crossover to create children.
    #    - Ensure size is 32. If not, fix by adding/removing random elements.
    # 5. Mutation: Swap a random element.
    # 6. Evaluate children.
    # 7. Replace worst in population.
    # 8. Repeat until time limit.
    
    # Why GA? It explores the space more broadly than single-trajectory hill climbing.
    # The "cross" operator specifically mixes features from two good parents,
    # potentially combining the "dense cluster" of one with the "spread out" nature of another.
    
    def _score_from_bitset(bitset, max_val=512):
        count = 0
        # We can vectorize this slightly better by using numpy if we convert, 
        # but Python int bitwise is very fast.
        # Let's stick to Python int for the core loop speed.
        for d in range(1, max_val // 2 + 1):
            if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                count += 1
        return count

    def _bitset_to_list(bitset):
        res = []
        b = bitset
        idx = 0
        while b:
            if b & 1:
                res.append(idx + 1)
            b >>= 1
            idx += 1
        return res

    def _list_to_bitset(lst):
        b = 0
        for x in lst:
            b |= (1 << (x - 1))
        return b

    def _random_bitset(n, max_val=512):
        indices = random.sample(range(max_val), n)
        b = 0
        for i in indices:
            b |= (1 << i)
        return b

    def _fix_size(bitset, target_n, max_val=512):
        # If bitset has too many bits, remove random ones.
        # If too few, add random ones.
        count = bitset.bit_count()
        if count == target_n:
            return bitset
        
        if count > target_n:
            # Remove (count - target_n) random bits
            bits_to_remove = count - target_n
            # Get list of indices
            indices = _bitset_to_list(bitset)
            to_remove = random.sample(indices, bits_to_remove)
            for idx in to_remove:
                bitset &= ~(1 << (idx - 1))
        else:
            # Add (target_n - count) random bits
            bits_to_add = target_n - count
            # Get current indices
            current_indices = set(_bitset_to_list(bitset))
            available = [i for i in range(1, max_val + 1) if i not in current_indices]
            to_add = random.sample(available, bits_to_add)
            for idx in to_add:
                bitset |= (1 << (idx - 1))
        
        return bitset

    def _crossover(parent1, parent2, n_target, max_val=512):
        # Uniform crossover: for each bit position, pick from p1 or p2 with 50% chance
        child = 0
        # We need to ensure the child has exactly n_target bits.
        # Simple uniform crossover on bitsets might result in arbitrary size.
        # Strategy: 
        # 1. Convert to sets of indices.
        # 2. For each index in 1..max_val, decide if it's in child.
        #    - If in both parents: high prob include.
        #    - If in one: medium prob.
        #    - If in none: low prob.
        # 3. Then fix size.
        
        # More efficient: 
        # Take n_target/2 bits from p1 and n_target/2 from p2?
        # This is "splice" or "half-half".
        
        idx1 = _bitset_to_list(parent1)
        idx2 = _bitset_to_list(parent2)
        
        # Shuffle
        random.shuffle(idx1)
        random.shuffle(idx2)
        
        half = n_target // 2
        child_indices = idx1[:half] + idx2[:n_target - half]
        
        # There might be duplicates if they share elements.
        # Remove duplicates
        child_set = set(child_indices)
        if len(child_set) < n_target:
            # Add random elements to fill
            current_bits = _list_to_bitset(list(child_set))
            current_indices = set(_bitset_to_list(current_bits))
            available = [i for i in range(1, max_val + 1) if i not in current_indices]
            needed = n_target - len(child_set)
            to_add = random.sample(available, needed)
            child_set.update(to_add)
        
        # If len > n_target (shouldn't happen with this logic, but just in case)
        if len(child_set) > n_target:
            child_list = list(child_set)
            random.shuffle(child_list)
            child_set = set(child_list[:n_target])
            
        return _list_to_bitset(list(child_set))

    def _mutate(bitset, n_target, max_val=512):
        # Swap a random element
        indices = _bitset_to_list(bitset)
        if not indices:
            return bitset
        
        rem_idx = random.choice(indices)
        add_idx = random.randint(1, max_val)
        
        # Ensure add_idx is not in set
        while add_idx in indices:
            add_idx = random.randint(1, max_val)
            
        new_b = bitset
        new_b &= ~(1 << (rem_idx - 1))
        new_b |= (1 << (add_idx - 1))
        return new_b

    # Initialize Population
    pop_size = 60
    population = []
    scores = []
    
    # Add some structured starts
    # 1. Random
    for _ in range(pop_size - 10):
        b = _random_bitset(n_target, max_val)
        population.append(b)
        
    # 2. Intervals
    for _ in range(5):
        start = random.randint(0, max_val - n_target)
        b = 0
        for i in range(start, start + n_target):
            b |= (1 << i)
        population.append(b)
        
    # 3. Modular (e.g., step 3)
    for _ in range(5):
        step = random.choice([2, 3, 4, 5])
        base = random.randint(0, step - 1)
        b = 0
        count = 0
        for i in range(max_val):
            if (i - base) % step == 0:
                b |= (1 << i)
                count += 1
                if count == n_target:
                    break
        if count == n_target:
            population.append(b)
        else:
            population.append(_random_bitset(n_target, max_val))
            
    # Evaluate initial population
    for b in population:
        s = _score_from_bitset(b)
        scores.append(s)
        
    best_score = max(scores)
    best_bitset = population[scores.index(best_score)]
    
    # GA Loop
    elite_size = 10
    
    while time.time() < deadline - 5.0:
        # Sort population by score
        paired = list(zip(scores, population))
        paired.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if paired[0][0] > best_score:
            best_score = paired[0][0]
            best_bitset = paired[0][1]
            
        # Selection: Keep top elites
        elites = [p for s, p in paired[:elite_size]]
        
        # Create new population
        new_population = elites[:]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Select parents (Tournament selection or Roulette)
            # Simple: Pick 2 random from top 30%
            top_30 = paired[:max(1, len(paired) // 3)]
            p1, s1 = random.choice(top_30)
            p2, s2 = random.choice(top_30)
            
            # Crossover
            if random.random() < 0.7:
                child = _crossover(p1, p2, n_target, max_val)
            else:
                # Clone one
                child = p1 if s1 > s2 else p2
                
            # Mutation
            if random.random() < 0.8:
                child = _mutate(child, n_target, max_val)
                
            new_population.append(child)
            
        # Evaluate new population
        # To save time, we can evaluate in parallel? No, standard python.
        # But we can stop early if time is tight.
        new_scores = []
        for b in new_population:
            if time.time() > deadline - 2.0:
                break
            s = _score_from_bitset(b)
            new_scores.append(s)
            
        if len(new_scores) < pop_size:
            # Pad with previous scores if we ran out of time? 
            # Or just break.
            break
            
        # Combine and select for next generation
        all_paired = list(zip(new_scores, new_population))
        # Sort and keep best pop_size
        all_paired.sort(key=lambda x: x[0], reverse=True)
        
        population = [p for s, p in all_paired[:pop_size]]
        scores = [s for s, p in all_paired[:pop_size]]
        
        # Update global best
        if scores[0] > best_score:
            best_score = scores[0]
            best_bitset = population[0]
            
    # Final Polish: Hill climb on the best found
    if time.time() < deadline - 2.0 and best_bitset:
        current_b = best_bitset
        current_score = best_score
        no_improve = 0
        max_no_improve = 500
        
        while no_improve < max_no_improve and time.time() < deadline - 1.0:
            # Try a few random swaps
            improved = False
            for _ in range(10):
                nb = _mutate(current_b, n_target, max_val)
                ns = _score_from_bitset(nb)
                if ns > current_score:
                    current_b = nb
                    current_score = ns
                    improved = True
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_b
                    break
            if not improved:
                no_improve += 1
                
    if best_bitset == 0:
        return list(range(1, 33))
        
    return _bitset_to_list(best_bitset)

# EVOLVE-BLOCK-END
