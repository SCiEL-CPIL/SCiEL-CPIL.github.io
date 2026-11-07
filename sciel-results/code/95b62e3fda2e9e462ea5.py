"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 128
    max_val = n

    # Precompute pairs for sum cases
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))

    def is_valid_addition(x, mask):
        """
        Check if adding x to set represented by mask is valid.
        mask: bitset where bit i-1 is 1 if i is in S.
        """
        if mask & (1 << (x - 1)):
            return True

        # Case 1: x acts as a member 'a' in a triple (a,b,c)
        # We need to ensure there are no distinct b, c in S such that
        # x+b in S, x+c in S, and b+c in S.
        
        # U_x = { y in S | x+y in S }
        U_x = mask & (mask >> x)
        
        # Iterate through bits in U_x. These are potential b's.
        m = U_x
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            
            # For this b, we need to find c in U_x, c != b, such that b+c in S.
            # V_b = { c in S | b+c in S } = mask & (mask >> b_val)
            V_b = mask & (mask >> b_val)
            
            # We need c in U_x and c in V_b and c != b.
            common = U_x & V_b
            if common & ~lsb_b:
                return False

        # Case 2: x acts as a sum a+b = x, with a,b in S.
        # We need to ensure there is no c in S \ {a,b} such that a+c in S and b+c in S.
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            
            # V_a = { c in S | a+c in S }
            V_a = mask & (mask >> a)
            # V_b = { c in S | b+c in S }
            V_b = mask & (mask >> b)
            
            common = V_a & V_b
            # Remove a and b from common
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False

        return True

    def popcount(x):
        return bin(x).count('1')

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    def try_extend(mask):
        """Try to greedily add elements to mask. Returns the extended mask."""
        changed = True
        while changed:
            changed = False
            for x in range(1, max_val + 1):
                if not (mask & (1 << (x - 1))):
                    if is_valid_addition(x, mask):
                        mask |= (1 << (x - 1))
                        changed = True
        return mask

    def is_valid_full(mask):
        """Check if the entire set represented by mask is valid."""
        # We can do this by trying to "re-add" elements? No, that's expensive.
        # Better: Check all triples?
        # A set is invalid if there exist distinct a,b,c in S such that a+b, a+c, b+c in S.
        # Iterate over all a,b,c? 128^3 is too slow.
        
        # Efficient check:
        # For each pair (a,b) in S with a<b:
        #   If a+b in S:
        #     Then we need to check if there exists c in S, c!=a, c!=b such that a+c in S and b+c in S.
        #     Let V_a = {c in S | a+c in S}
        #     Let V_b = {c in S | b+c in S}
        #     If (V_a & V_b) has any bit other than a or b, then invalid.
        
        for a in range(1, max_val + 1):
            if not (mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, max_val + 1):
                if not (mask & (1 << (b - 1))):
                    continue
                s = a + b
                if s > max_val:
                    break
                if not (mask & (1 << (s - 1))):
                    continue
                
                # Check for c
                V_a = mask & (mask >> a)
                V_b = mask & (mask >> b)
                common = V_a & V_b
                # Remove a and b
                a_bit = 1 << (a - 1)
                b_bit = 1 << (b - 1)
                if common & ~(a_bit | b_bit):
                    return False
        return True

    # ---- Structural start ----
    # Start with the known good structure: L U H
    K = 64
    H = list(range(K, max_val + 1))                 # {64..128}
    max_L = (K - 1) // 2                            # 31
    start_L = max_L // 2 + 1                        # 16
    L = list(range(start_L, max_L + 1))             # {16..31}

    best_mask = 0
    for v in H + L:
        best_mask |= (1 << (v - 1))
    
    # Verify and extend the structural start
    if is_valid_full(best_mask):
        best_mask = try_extend(best_mask)
    
    best_size = popcount(best_mask)

    # GA Parameters
    population_size = 40
    generations = 200  # We will stop earlier if time runs out

    # Initialize Population
    # 1. The best mask found so far
    # 2. Random masks
    # 3. Mutations of the best mask
    
    population = []
    
    # Add the structural best
    population.append(best_mask)
    
    # Add random masks
    random.seed(42)
    for _ in range(population_size - 1):
        # Random density ~0.5
        mask = 0
        for i in range(1, max_val + 1):
            if random.random() < 0.5:
                mask |= (1 << (i - 1))
        # Repair: Try to extend? No, extend might not be valid if the base is invalid.
        # Actually, the "repair" strategy in the prompt was:
        # 1. Generate random mask
        # 2. Check validity. If invalid, repair.
        # 3. Extend greedily.
        
        # Let's implement a simple repair:
        # While invalid:
        #   Find a violation (a,b,c,a+b,a+c,b+c).
        #   Remove one of {a,b,c,a+b,a+c,b+c} at random.
        
        def find_violation(mask):
            """Returns (a,b,c) if valid, else None. Or returns the set of elements to remove."""
            # To be fast, we can just return the first violation found.
            # The violation is the set {a,b,c,a+b,a+c,b+c}.
            for a in range(1, max_val + 1):
                if not (mask & (1 << (a - 1))):
                    continue
                for b in range(a + 1, max_val + 1):
                    if not (mask & (1 << (b - 1))):
                        continue
                    s = a + b
                    if s > max_val:
                        break
                    if not (mask & (1 << (s - 1))):
                        continue
                    
                    V_a = mask & (mask >> a)
                    V_b = mask & (mask >> b)
                    common = V_a & V_b
                    a_bit = 1 << (a - 1)
                    b_bit = 1 << (b - 1)
                    check = common & ~(a_bit | b_bit)
                    if check:
                        # Find c
                        c_bit = check & -check
                        c_val = c_bit.bit_length()
                        # The violation involves: a, b, c, a+b, a+c, b+c
                        # We return these values to remove one
                        return [a, b, c_val, a+b, a+c_val, b+c_val]
            return None

        def repair(mask):
            # Iterate until valid or max steps
            steps = 0
            while steps < 100:
                viol = find_violation(mask)
                if viol is None:
                    break
                # Remove a random element from the violation
                # Note: Some elements in 'viol' might not be in mask?
                # a,b,c are in mask. a+b is in mask.
                # a+c and b+c might be > 128 or not in mask?
                # Wait, the check in find_violation ensures a+b is in mask.
                # It ensures c is in S and a+c in S and b+c in S.
                # So all 6 are in mask.
                
                # Remove one
                elem_to_remove = random.choice(viol)
                mask &= ~(1 << (elem_to_remove - 1))
                steps += 1
            return mask

        # Apply repair to random mask
        mask = repair(mask)
        # Then extend
        mask = try_extend(mask)
        population.append(mask)

    # Evaluate initial population
    scores = [popcount(m) for m in population]
    
    best_overall_mask = population[0]
    best_overall_score = scores[0]
    for i in range(1, len(population)):
        if scores[i] > best_overall_score:
            best_overall_score = scores[i]
            best_overall_mask = population[i]
    
    if best_overall_score > best_size:
        best_size = best_overall_score
        best_mask = best_overall_mask

    # GA Loop
    for gen in range(generations):
        if time.time() > deadline - 5.0:
            break
            
        # Sort population by score
        indexed_pop = list(zip(scores, population))
        indexed_pop.sort(key=lambda x: x[0], reverse=True)
        scores = [x[0] for x in indexed_pop]
        population = [x[1] for x in indexed_pop]
        
        # Update best
        if scores[0] > best_size:
            best_size = scores[0]
            best_mask = population[0]

        # Selection: Elitism (keep top 10) + Tournament (fill rest)
        elites = population[:10]
        new_population = elites[:]
        
        while len(new_population) < population_size:
            # Tournament selection
            i = random.randint(0, len(population)-1)
            j = random.randint(0, len(population)-1)
            if scores[i] >= scores[j]:
                parent = population[i]
            else:
                parent = population[j]
            
            # Mutation
            # Flip 1-3 bits
            num_flips = random.randint(1, 3)
            child = parent
            for _ in range(num_flips):
                bit = random.randint(1, max_val)
                child ^= (1 << (bit - 1))
            
            # Crossover?
            # The prompt suggested crossover.
            # Let's do uniform crossover with another parent 50% of the time.
            if random.random() < 0.5:
                # Pick another parent
                k = random.randint(0, len(population)-1)
                other = population[k]
                # Uniform crossover
                child = 0
                for i in range(1, max_val + 1):
                    bit = 1 << (i - 1)
                    if random.random() < 0.5:
                        if parent & bit:
                            child |= bit
                    else:
                        if other & bit:
                            child |= bit
                
                # Mutation after crossover
                num_flips = random.randint(1, 3)
                for _ in range(num_flips):
                    bit = random.randint(1, max_val)
                    child ^= (1 << (bit - 1))

            # Repair and Extend
            child = repair(child)
            child = try_extend(child)
            new_population.append(child)

        # Evaluate new population
        scores = [popcount(m) for m in new_population]
        population = new_population

    # Final Polish on Best
    if time.time() < deadline - 1.0:
        final_mask = try_extend(best_mask)
        if popcount(final_mask) > best_size:
            best_mask = final_mask
            best_size = popcount(final_mask)

    # Convert best_mask to list of ints
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)

# EVOLVE-BLOCK-END
