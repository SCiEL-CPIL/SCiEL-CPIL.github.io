import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    N = 32
    MAX_VAL = 512
    random.seed(12345)

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_val, new_val):
        """
        Calculate the change in score when replacing old_val with new_val in the set represented by mask.
        Returns the delta (new_score - old_score).
        """
        old_bit = 1 << old_val
        new_bit = 1 << new_val
        new_mask = mask ^ old_bit ^ new_bit
        
        delta = 0
        # Only check d where the status might change
        # Actually, checking all 256 is fast enough and safer.
        for d in range(1, 256):
            # Check if AP with diff d existed in old mask
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            # Check if AP with diff d exists in new mask
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                if new_has:
                    delta += 1
                else:
                    delta -= 1
        return delta

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def gen_structured_candidates():
        cands = []
        # Interval
        lst = list(range(1, 33))
        m = list_to_mask(lst)
        cands.append((lst, m, score_mask(m)))
        
        # APs
        for start in range(1, 20):
            for step in range(3, 20):
                lst = [start + i * step for i in range(N)]
                if max(lst) > MAX_VAL:
                    continue
                m = list_to_mask(lst)
                cands.append((lst, m, score_mask(m)))
        
        # Two interleaved APs
        for step1 in range(3, 15):
            for step2 in range(3, 15):
                if step1 == step2:
                    continue
                lst = []
                for i in range(16):
                    lst.append(1 + i * step1)
                    lst.append(2 + i * step2)
                if len(set(lst)) != N or max(lst) > MAX_VAL:
                    continue
                m = list_to_mask(lst)
                cands.append((lst, m, score_mask(m)))
                
        # Sumset A + k*B
        for mult in [2, 3, 4, 5]:
            for trial in range(100):
                sizeA = random.randint(3, 8)
                sizeB = random.randint(3, 8)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    # Pick N elements from S
                    # Prefer elements that are part of more APs in S? 
                    # For now, just random sample or sorted
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    cands.append((lst, m, score_mask(m)))
        return cands

    def gen_random_candidates(count):
        cands = []
        for _ in range(count):
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            cands.append((lst, m, score_mask(m)))
        return cands

    def local_search(lst, mask, score, max_iter=50):
        """
        Perform hill climbing on a single individual.
        """
        s_set = set(lst)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 1.0:
            improved = False
            iter_count += 1
            # Try to improve each position
            for i in range(N):
                old_val = lst[i]
                best_new_val = old_val
                best_score = score
                # Try a few random replacements
                for _ in range(10):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in s_set:
                        continue
                    d = delta_score(mask, old_val, new_val)
                    new_score = score + d
                    if new_score > best_score:
                        best_score = new_score
                        best_new_val = new_val
                
                if best_new_val != old_val:
                    # Apply the best move found for this position
                    s_set.remove(old_val)
                    s_set.add(best_new_val)
                    lst[i] = best_new_val
                    mask ^= (1 << old_val) ^ (1 << best_new_val)
                    score = best_score
                    improved = True
                    # Restart loop to allow other positions to be improved with new context
                    # Or just continue to next i? Restarting is safer for steepest ascent.
                    break 
        return lst, mask, score

    # Initialize population
    pop_size = 50
    population = []
    
    # Add structured candidates
    structured = gen_structured_candidates()
    population.extend(structured)
    
    # Add random candidates
    random_cands = gen_random_candidates(pop_size - len(structured))
    population.extend(random_cands)
    
    # Ensure population size
    while len(population) < pop_size:
        lst = random.sample(range(1, MAX_VAL + 1), N)
        m = list_to_mask(lst)
        population.append((lst, m, score_mask(m)))
        
    # Sort by fitness
    population.sort(key=lambda x: x[2], reverse=True)
    
    best_list = population[0][0][:]
    best_mask = population[0][1]
    best_score = population[0][2]
    
    print(f"Initial best: {best_score}", file=sys.stderr)
    
    # GA Loop
    generation = 0
    no_improve_count = 0
    elite_count = 5
    
    while time.time() < deadline - 10.0: # Leave 10s for final polish
        generation += 1
        if generation % 20 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)
        
        new_population = []
        
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            # Simple tournament: pick 3 random, take best
            candidates = [random.choice(population) for _ in range(3)]
            parent = max(candidates, key=lambda x: x[2])
            
            # Mutation / Crossover
            if random.random() < 0.5:
                # Crossover: Uniform
                child_lst = []
                seen = set()
                for i in range(N):
                    if random.random() < 0.5:
                        val = parent[0][i]
                    else:
                        # Pick from another random parent or random value
                        other = random.choice(population)
                        val = other[0][i]
                    
                    if val not in seen:
                        child_lst.append(val)
                        seen.add(val)
                    else:
                        # Collision, pick random
                        while True:
                            val = random.randint(1, MAX_VAL)
                            if val not in seen:
                                child_lst.append(val)
                                seen.add(val)
                                break
            else:
                # Mutation: Start with parent, mutate a few elements
                child_lst = parent[0][:]
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    idx = random.randint(0, N-1)
                    old_val = child_lst[idx]
                    new_val = random.randint(1, MAX_VAL)
                    if new_val != old_val and new_val not in child_lst:
                        child_lst[idx] = new_val
            
            child_mask = list_to_mask(child_lst)
            child_score = score_mask(child_mask)
            
            # Local search on child
            child_lst, child_mask, child_score = local_search(child_lst, child_mask, child_score, max_iter=20)
            
            new_population.append((child_lst, child_mask, child_score))
            
            if child_score > best_score:
                best_score = child_score
                best_list = child_lst[:]
                best_mask = child_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Update population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # If no improvement for many generations, inject diversity
        if no_improve_count > 100:
            # Add some random individuals
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                population.append((lst, m, score_mask(m)))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve_count = 0

    # Final Polish: Exhaustive local search on the best individual
    current_lst = best_list[:]
    current_mask = best_mask
    current_score = best_score
    current_set = set(current_lst)
    
    print(f"Starting final polish with score {current_score}", file=sys.stderr)
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(N):
            old_val = current_lst[i]
            best_new_val = old_val
            best_new_score = current_score
            
            # Try all possible new values
            for new_val in range(1, MAX_VAL + 1):
                if new_val == old_val or new_val in current_set:
                    continue
                d = delta_score(current_mask, old_val, new_val)
                new_score = current_score + d
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
            
            if best_new_val != old_val:
                # Apply move
                current_set.remove(old_val)
                current_set.add(best_new_val)
                current_lst[i] = best_new_val
                current_mask ^= (1 << old_val) ^ (1 << best_new_val)
                current_score = best_new_score
                improved = True
                # Restart loop to ensure we don't miss improvements due to order
                break
                
    print(f"Final score: {current_score}", file=sys.stderr)
    
    return current_lst

# EVOLVE-BLOCK-END
