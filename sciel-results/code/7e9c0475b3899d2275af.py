import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets
    
    # Precompute popcount for masks (for debugging or alternative counting, but we use int.bit_count)
    # Actually, we just need bit_count on big integers.
    
    # Precompute comparability bitsets
    # comp_bits[i] is a big integer where bit j is 1 if mask i is comparable with mask j (i != j)
    # Note: Strict subset means i != j and (i & j) == i or (i & j) == j.
    comp_bits = [0] * total_masks
    
    # Optimization: Group masks by popcount to speed up precomputation? 
    # Or just do it directly. 4096^2 is 16 million operations. In Python, this might be slow.
    # Let's use a faster method:
    # For each mask i, its comparable sets are all its subsets (except itself) and all its supersets (except itself).
    # We can generate these efficiently.
    
    # Helper: get all subsets of a mask
    def get_subsets(mask):
        subs = []
        sub = mask
        while sub:
            subs.append(sub)
            sub = (sub - 1) & mask
        return subs # Does not include 0
    
    # Helper: get all supersets of a mask
    def get_supersets(mask):
        # Supersets of mask are masks s such that (s & mask) == mask
        # This is equivalent to: s = mask | x where x is a subset of (~mask) & ((1<<n)-1)
        free_bits = ((1 << n) - 1) ^ mask
        sups = []
        sub = free_bits
        while True:
            s = mask | sub
            if s != mask:
                sups.append(s)
            if sub == 0:
                break
            sub = (sub - 1) & free_bits
        return sups

    # Precomputing comp_bits for all 4096 masks
    # This loop might take a few seconds. Let's optimize.
    # Instead of iterating all pairs, we can build the bitset for each i.
    for i in range(total_masks):
        bits = 0
        # Add subsets
        for s in get_subsets(i):
            bits |= (1 << s)
        # Add supersets
        for s in get_supersets(i):
            bits |= (1 << s)
        comp_bits[i] = bits
        
    # Function to calculate fitness (number of comparable pairs)
    def calc_fitness(family_bits):
        # family_bits is a big int representing the set of masks in the family
        # Count edges: sum over i in family of (comp_bits[i] & family_bits).bit_count()
        # Since each edge is counted twice (once for each endpoint), divide by 2.
        # Iterate over set bits in family_bits
        count = 0
        f = family_bits
        while f:
            # Get lowest set bit index
            lsb = f & -f
            idx = lsb.bit_length() - 1
            count += (comp_bits[idx] & family_bits).bit_count()
            f ^= lsb
        return count // 2

    # Initialize population
    pop_size = 100
    population = []
    
    # Seed 1: Greedy chain-based
    random.seed(42)
    chain = [(1 << size) - 1 for size in range(13)]
    fam_bits = 0
    for mask in chain:
        fam_bits |= (1 << mask)
    # Add random ones to reach 40
    while fam_bits.bit_count() < m:
        c = random.randint(0, total_masks - 1)
        if not (fam_bits >> c) & 1:
            fam_bits |= (1 << c)
    population.append(fam_bits)
    
    # Seed 2: Random
    for _ in range(pop_size - 1):
        fam_bits = 0
        # Randomly pick 40 distinct masks
        masks = random.sample(range(total_masks), m)
        for mask in masks:
            fam_bits |= (1 << mask)
        population.append(fam_bits)
    
    # Evaluate initial population
    fitnesses = [calc_fitness(f) for f in population]
    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_family = population[best_idx]
    
    # Genetic Algorithm Loop
    generation = 0
    no_improve_gen = 0
    max_no_improve = 20
    
    while time.time() < deadline - 2.0:
        generation += 1
        new_population = []
        
        # Elitism: keep top 10%
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            if fitnesses[p1_idx] < fitnesses[p2_idx]:
                p1_idx, p2_idx = p2_idx, p1_idx
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover: Bitwise mix
            # Create a random mask to select bits from p1 or p2
            # We want to keep exactly 40 bits set.
            # Simple method: Union, then trim. Or Intersect, then add.
            # Let's try: Start with p1, swap some bits with p2.
            child = p1
            # Pick 5-10 bits to swap
            swaps = random.randint(5, 10)
            for _ in range(swaps):
                # Pick a bit from child to remove
                if child.bit_count() == 0:
                    break
                lsb_c = child & -child
                idx_c = lsb_c.bit_length() - 1
                # Pick a bit from p2 to add (that is not in child)
                avail = p2 & ~child
                if avail == 0:
                    break
                lsb_a = avail & -avail
                idx_a = lsb_a.bit_length() - 1
                child ^= lsb_c
                child |= lsb_a
            
            # Mutation: Random bit flip
            if random.random() < 0.5:
                # Remove one random bit
                lsb_c = child & -child
                child ^= lsb_c
                # Add one random bit
                avail = (~child) & ((1 << total_masks) - 1)
                if avail:
                    lsb_a = avail & -avail
                    child |= lsb_a
            
            # Ensure exactly 40 bits
            if child.bit_count() != m:
                # Adjust
                while child.bit_count() < m:
                    avail = (~child) & ((1 << total_masks) - 1)
                    if not avail: break
                    lsb_a = avail & -avail
                    child |= lsb_a
                while child.bit_count() > m:
                    lsb_c = child & -child
                    child ^= lsb_c
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [calc_fitness(f) for f in population]
        
        # Local Search on top candidates
        top_k = 10
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        
        for i in range(top_k):
            if time.time() > deadline - 2.0:
                break
            idx = sorted_indices[i]
            fam = population[idx]
            fit = fitnesses[idx]
            
            # Single move local search
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Try all possible single bit swaps
                # To be fast, we can just try a random subset of swaps
                # Or iterate through all bits in family and try swapping with a random non-family bit
                for _ in range(100): # Limit swaps per local search step
                    lsb_c = fam & -fam
                    idx_c = lsb_c.bit_length() - 1
                    
                    avail = (~fam) & ((1 << total_masks) - 1)
                    if not avail:
                        break
                    # Pick random available bit
                    # To pick random bit from big int:
                    # Convert to string? No, too slow.
                    # Use random index and check?
                    # Better: use random bits from the avail mask.
                    # We can pick a random integer 0..total_masks-1 and check if it's in avail.
                    # Or, since avail is sparse, we can just try a few random candidates.
                    for _ in range(5):
                        idx_a = random.randint(0, total_masks - 1)
                        if (avail >> idx_a) & 1:
                            break
                    else:
                        continue
                    
                    lsb_a = 1 << idx_a
                    new_fam = fam ^ lsb_c | lsb_a
                    
                    new_fit = calc_fitness(new_fam)
                    if new_fit > fit:
                        fam = new_fam
                        fit = new_fit
                        improved = True
                        # Update in population
                        population[idx] = fam
                        fitnesses[idx] = fit
            
            if fit > best_fitness:
                best_fitness = fit
                best_family = fam
                no_improve_gen = 0
            else:
                no_improve_gen += 1
        
        if no_improve_gen > max_no_improve:
            # Perturb best solution
            fam = best_family
            for _ in range(5):
                lsb_c = fam & -fam
                idx_c = lsb_c.bit_length() - 1
                avail = (~fam) & ((1 << total_masks) - 1)
                if not avail: break
                idx_a = random.randint(0, total_masks - 1)
                while not ((avail >> idx_a) & 1):
                    idx_a = random.randint(0, total_masks - 1)
                lsb_a = 1 << idx_a
                fam ^= lsb_c | lsb_a
            population[random.randint(0, pop_size - 1)] = fam
            fitnesses[random.randint(0, pop_size - 1)] = calc_fitness(fam)
            no_improve_gen = 0

    # Final Local Search on best family
    fam = best_family
    fit = best_fitness
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        for _ in range(200):
            lsb_c = fam & -fam
            idx_c = lsb_c.bit_length() - 1
            avail = (~fam) & ((1 << total_masks) - 1)
            if not avail:
                break
            for _ in range(5):
                idx_a = random.randint(0, total_masks - 1)
                if (avail >> idx_a) & 1:
                    break
            else:
                continue
            lsb_a = 1 << idx_a
            new_fam = fam ^ lsb_c | lsb_a
            new_fit = calc_fitness(new_fam)
            if new_fit > fit:
                fam = new_fam
                fit = new_fit
                improved = True

    # Convert best_family bitset to list of lists
    result = []
    f = fam
    while f:
        lsb = f & -f
        idx = lsb.bit_length() - 1
        # Convert mask idx to list of ints
        s = []
        for b in range(n):
            if (idx >> b) & 1:
                s.append(b)
        result.append(s)
        f ^= lsb
        
    # Ensure types are correct
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
        
    return result

# Test run
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:5])
