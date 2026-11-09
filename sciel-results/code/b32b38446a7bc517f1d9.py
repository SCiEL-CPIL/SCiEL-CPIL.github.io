import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that a+2d <= 512 for some a >= 1, i.e., 2d <= 511, d <= 255
    random.seed(42)

    # Precompute for each d in 1..255, the list of valid a values
    # For a given d, valid a: 1 <= a and a + 2d <= 512, so a <= 512 - 2d
    # Number of valid a for difference d: max(0, 512 - 2d)
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask for some a."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        lst = []
        for x in range(1, MAX_VAL + 1):
            if mask & (1 << x):
                lst.append(x)
        return lst

    # Delta scoring: when we remove element x and add element y,
    # for each d, we need to check if the AP count changes.
    # 
    # For a given d, the AP (a, a+d, a+2d) is "covered" if all three are in the set.
    # When we remove x, an AP that was covered might become uncovered if x was the last
    # element keeping it covered. Similarly when we add y.
    #
    # Key insight: for a fixed d, the number of APs is either 0 or >= 1 (we just need
    # at least one). So the score contribution for d is 1 if ANY AP with difference d
    # is fully in the set, else 0.
    #
    # When removing x: for each d, if x is part of some AP with difference d, and that
    # AP was the ONLY AP with difference d fully in the set, then we lose d.
    # When adding y: for each d, if y completes an AP with difference d that wasn't
    # previously complete, we gain d.
    #
    # For efficiency, let's precompute for each element x and each d, the set of a values
    # such that x is in {a, a+d, a+2d} and the AP is valid.
    #
    # Actually, a simpler approach for delta scoring:
    # - Precompute for each d, the list of all valid APs (triples)
    # - Maintain a count for each d: how many APs with difference d are fully in the set
    # - Score = number of d with count[d] > 0
    # - When adding/removing an element, update counts for affected d's

    # Precompute: for each d, list of AP triples (a, a+d, a+2d)
    # And for each element x, which (d, position) pairs it participates in
    # position: 0 for a, 1 for a+d, 2 for a+2d

    # For delta scoring, we need:
    # For each x and d: the list of a values where x is in the AP (a, a+d, a+2d)
    # x = a => a = x
    # x = a+d => a = x - d  
    # x = a+2d => a = x - 2d
    # Valid if 1 <= a and a + 2d <= 512

    # Let's precompute for each x (1..512) and d (1..255):
    # ap_a_values[x][d] = list of a values where x participates in AP with diff d
    # And for each (x, d, a), we can determine the other two elements

    # For the local search, we'll maintain:
    # - The current set as a boolean array or mask
    # - For each d, the count of APs fully in the set
    # - The score = number of d with count > 0

    def init_counts(mask):
        """Initialize AP counts for each d given a mask."""
        counts = [0] * (MAX_D + 1)
        for d in range(1, MAX_D + 1):
            cnt = 0
            max_a = MAX_VAL - 2 * d
            if max_a < 1:
                continue
            for a in range(1, max_a + 1):
                if (mask >> a) & 1 and (mask >> (a + d)) & 1 and (mask >> (a + 2 * d)) & 1:
                    cnt += 1
            counts[d] = cnt
        return counts

    def compute_score_from_counts(counts):
        """Compute score from AP counts."""
        return sum(1 for d in range(1, MAX_D + 1) if counts[d] > 0)

    def get_aps_for_element(x, d):
        """Get all a values where element x is in AP (a, a+d, a+2d) with difference d."""
        a_vals = []
        # x = a
        if 1 <= x and x + 2 * d <= MAX_VAL:
            a_vals.append(x)
        # x = a + d
        a = x - d
        if 1 <= a and a + 2 * d <= MAX_VAL:
            a_vals.append(a)
        # x = a + 2d
        a = x - 2 * d
        if 1 <= a and a + 2 * d <= MAX_VAL:
            a_vals.append(a)
        return a_vals

    def delta_score_add_remove(mask, counts, remove_x, add_y):
        """
        Compute new score after removing remove_x and adding add_y.
        Returns (new_score, new_counts) or just new_score if we don't need new_counts.
        """
        if remove_x == add_y:
            return compute_score_from_counts(counts), counts
        
        # First, handle removal of remove_x
        # For each d, check if removing remove_x decreases any AP count to 0
        # An AP (a, a+d, a+2d) with difference d contains remove_x if:
        #   remove_x = a, or remove_x = a+d, or remove_x = a+2d
        
        new_counts = counts[:]
        
        # Process removal
        for d in range(1, MAX_D + 1):
            a_vals = get_aps_for_element(remove_x, d)
            for a in a_vals:
                # Check if this AP was fully in the set (before removal)
                a1, a2, a3 = a, a + d, a + 2 * d
                # All three must be in mask, and at least one is remove_x
                # Since we know remove_x is in this AP, check the other two
                other1 = a1 if a1 != remove_x else None
                other2 = a2 if a2 != remove_x else None
                other3 = a3 if a3 != remove_x else None
                
                # Check if all three were in the mask
                all_in = True
                for val in [a1, a2, a3]:
                    if val != remove_x and not (mask >> val) & 1:
                        all_in = False
                        break
                if all_in:
                    # This AP was fully in the set and contains remove_x
                    new_counts[d] -= 1
        
        # Process addition of add_y
        for d in range(1, MAX_D + 1):
            a_vals = get_aps_for_element(add_y, d)
            for a in a_vals:
                a1, a2, a3 = a, a + d, a + 2 * d
                # Check if the other two elements (besides add_y) are in the mask
                # But we need to be careful: if remove_x was one of them, it's no longer there
                all_in = True
                for val in [a1, a2, a3]:
                    if val == add_y:
                        continue
                    if val == remove_x:
                        # This element was removed, so it's not in the set
                        all_in = False
                        break
                    if not (mask >> val) & 1:
                        all_in = False
                        break
                if all_in:
                    new_counts[d] += 1
        
        new_score = compute_score_from_counts(new_counts)
        return new_score, new_counts

    # Faster delta scoring: precompute for each x and d, the APs more efficiently
    # Let's use a different approach: for each d, precompute the list of APs
    # and for each AP, which elements it contains
    
    # Actually, let's just use the bit-parallel score for initial evaluation
    # and a smarter local search that evaluates moves more carefully

    def local_search(lst, max_iters=1000, time_limit=None):
        """Greedy local search with delta scoring."""
        lst = lst[:]
        mask = list_to_mask(lst)
        counts = init_counts(mask)
        s = compute_score_from_counts(counts)
        improved = True
        iter_count = 0
        seen_in_list = set(lst)
        
        while improved and iter_count < max_iters:
            if time_limit and time.time() > time_limit:
                break
            improved = False
            iter_count += 1
            for i in range(N):
                old_val = lst[i]
                best_new_val = None
                best_new_score = s
                best_new_counts = None
                
                # Try all possible new values
                for new_val in range(1, MAX_VAL + 1):
                    if new_val == old_val or new_val in seen_in_list:
                        continue
                    
                    # Use delta scoring
                    new_score, new_counts = delta_score_add_remove(mask, counts, old_val, new_val)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                        best_new_counts = new_counts
                
                if best_new_val is not None:
                    # Apply the move
                    lst[i] = best_new_val
                    seen_in_list.remove(old_val)
                    seen_in_list.add(best_new_val)
                    mask = mask ^ (1 << old_val) ^ (1 << best_new_val)
                    counts = best_new_counts
                    s = best_new_score
                    improved = True
                    break  # Restart the scan after a change
        
        return lst, s

    # Initialize population
    pop_size = 80
    population = []
    
    # Structured: intervals
    for offset in range(0, 481, 8):
        lst = list(range(offset + 1, offset + N + 1))
        if max(lst) <= MAX_VAL:
            m = list_to_mask(lst)
            s = score_mask(m)
            population.append((lst, m, s))
    
    # Structured: APs
    for step in range(5, 18):
        lst = [1 + i * step for i in range(N)]
        if max(lst) <= MAX_VAL:
            m = list_to_mask(lst)
            s = score_mask(m)
            population.append((lst, m, s))
    
    # Structured: two intervals
    for split in range(8, N - 7):
        for gap in range(1, 50):
            part1 = list(range(1, split + 1))
            start2 = split + 1 + gap
            part2 = list(range(start2, start2 + N - split + 1))
            if max(part2) <= MAX_VAL:
                lst = part1 + part2
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
    
    # Random
    while len(population) < pop_size:
        lst = random.sample(range(1, MAX_VAL + 1), N)
        m = list_to_mask(lst)
        s = score_mask(m)
        population.append((lst, m, s))
    
    population = population[:pop_size]
    population.sort(key=lambda x: -x[2])
    best_list = population[0][0][:]
    best_mask = population[0][1]
    best_score = population[0][2]
    
    print(f"Initial best score: {best_score}", file=sys.stderr)
    
    # Genetic algorithm loop
    gen = 0
    no_improve_gens = 0
    elite_size = 12
    
    while time.time() < deadline - 40:  # Leave 40s for final polish
        gen += 1
        if gen % 10 == 0:
            print(f"Gen {gen}, best={best_score}, top5={[p[2] for p in population[:5]]}", file=sys.stderr)
        
        # Selection: tournament
        def tournament_select(k=5):
            candidates = random.sample(population, min(k, len(population)))
            return max(candidates, key=lambda x: x[2])
        
        new_pop = []
        
        # Elitism
        for i in range(elite_size):
            new_pop.append(population[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 40:
                break
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Uniform crossover
            c1 = []
            c2 = []
            for i in range(N):
                if random.random() < 0.5:
                    c1.append(p1[0][i])
                    c2.append(p2[0][i])
                else:
                    c1.append(p2[0][i])
                    c2.append(p1[0][i])
            
            # Mutate
            for i in range(N):
                if random.random() < 0.2:
                    c1[i] = random.randint(1, MAX_VAL)
                if random.random() < 0.2:
                    c2[i] = random.randint(1, MAX_VAL)
            
            # Ensure distinctness
            for c in [c1, c2]:
                seen = set()
                valid = []
                for x in c:
                    if x not in seen and 1 <= x <= MAX_VAL:
                        seen.add(x)
                        valid.append(x)
                while len(valid) < N:
                    x = random.randint(1, MAX_VAL)
                    if x not in seen:
                        seen.add(x)
                        valid.append(x)
                c[:] = valid[:N]
            
            m1 = list_to_mask(c1)
            s1 = score_mask(m1)
            m2 = list_to_mask(c2)
            s2 = score_mask(m2)
            
            new_pop.append((c1, m1, s1))
            new_pop.append((c2, m2, s2))
        
        population = new_pop
        population.sort(key=lambda x: -x[2])
        
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve_gens = 0
        else:
            no_improve_gens += 1
        
        # If stuck, inject diversity
        if no_improve_gens > 20:
            for i in range(8):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population[i] = (lst, m, s)
            population.sort(key=lambda x: -x[2])
            no_improve_gens = 0
    
    # Final local search polish on best and top few
    print(f"Before polish: best={best_score}", file=sys.stderr)
    
    candidates_to_polish = [p[0][:] for p in population[:15]]
    
    for lst in candidates_to_polish:
        if time.time() > deadline - 10:
            break
        polished, s = local_search(lst, max_iters=500, time_limit=deadline - 10)
        if s > best_score:
            best_score = s
            best_list = polished
            best_mask = list_to_mask(polished)
            print(f"Polish improved: {best_score}", file=sys.stderr)
    
    # One more round of local search on the best with more iterations
    if time.time() < deadline - 5:
        polished, s = local_search(best_list, max_iters=2000, time_limit=deadline - 5)
        if s > best_score:
            best_score = s
            best_list = polished
            best_mask = list_to_mask(polished)
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list

# EVOLVE-BLOCK-END

