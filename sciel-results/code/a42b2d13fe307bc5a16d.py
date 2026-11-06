import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    N = 32
    RANGE = 512
    MAX_D = 255  # Max possible d: a=1, a+2d<=512 => 2d<=511 => d<=255

    # Precompute bit masks for each value 1..512
    # mask for value v is 1 << (v-1)
    val_masks = [1 << (v - 1) for v in range(RANGE + 1)]

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in set represented by mask."""
        count = 0
        # d ranges from 1 to 255
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= val_masks[x]
        return mask

    def greedy_polish(S_init, time_limit=10.0):
        """
        Local search: iteratively try to replace each element with a better one.
        Uses bitset scoring for speed.
        """
        S = list(S_init)
        mask = set_to_mask(S)
        current_score = score_mask(mask)
        s_set = set(S)
        
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for i in range(N):
                old_val = S[i]
                s_set.remove(old_val)
                mask &= ~val_masks[old_val]
                
                best_cand = None
                best_score = current_score
                
                # Try all candidates not in s_set
                # To speed up, we can sample if too slow, but 512 is small.
                # However, 32 * 512 * 255 bit ops per pass is ~4M ops.
                # In Python, this might be ~0.1-0.5s per pass.
                # So we can do several passes.
                
                for cand in range(1, RANGE + 1):
                    if cand in s_set:
                        continue
                    
                    new_mask = mask | val_masks[cand]
                    score = score_mask(new_mask)
                    
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand is not None:
                    S[i] = best_cand
                    s_set.add(best_cand)
                    mask |= val_masks[best_cand]
                    current_score = best_score
                    improved = True
                else:
                    # Restore
                    s_set.add(old_val)
                    mask |= val_masks[old_val]
        
        return S, current_score

    def simulated_annealing(S_init, time_limit):
        """
        SA with 1-replace moves.
        """
        S = list(S_init)
        mask = set_to_mask(S)
        current_score = score_mask(mask)
        best_S = list(S)
        best_score = current_score
        s_set = set(S)
        
        T = 5.0
        cooling = 0.9999
        min_T = 0.1
        
        start = time.time()
        
        while time.time() - start < time_limit and T > min_T:
            i = random.randint(0, N - 1)
            old_val = S[i]
            new_val = random.randint(1, RANGE)
            
            if new_val == old_val:
                continue
            if new_val in s_set:
                continue
            
            # Apply move
            s_set.remove(old_val)
            s_set.add(new_val)
            mask &= ~val_masks[old_val]
            mask |= val_masks[new_val]
            S[i] = new_val
            
            new_score = score_mask(mask)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = list(S)
            else:
                # Revert
                s_set.remove(new_val)
                s_set.add(old_val)
                mask &= ~val_masks[new_val]
                mask |= val_masks[old_val]
                S[i] = old_val
            
            T *= cooling
        
        return best_S, best_score

    def greedy_construct():
        """
        Construct a set by greedily adding elements that maximize distinct differences.
        """
        S = []
        s_set = set()
        mask = 0
        
        for _ in range(N):
            best_val = -1
            best_score = -1
            
            # To speed up, we don't need to check all 512 every time if we're smart,
            # but for the first few elements, it's important.
            # Let's just iterate all.
            # Shuffle to break ties and add diversity
            candidates = list(range(1, RANGE + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                if val in s_set:
                    continue
                
                new_mask = mask | val_masks[val]
                sc = score_mask(new_mask)
                
                if sc > best_score:
                    best_score = sc
                    best_val = val
            
            if best_val != -1:
                S.append(best_val)
                s_set.add(best_val)
                mask |= val_masks[best_val]
        
        return sorted(S)

    best_overall = None
    best_overall_score = 0

    # Phase 1: Generate diverse initializations
    inits = []
    
    # 1. Greedy
    g1 = greedy_construct()
    inits.append(g1)
    
    # 2. Random
    inits.append(random.sample(range(1, RANGE + 1), N))
    
    # 3. Structured: Spread out elements
    step = RANGE // N
    structured = [i * step + 1 for i in range(N)]
    inits.append(structured)
    
    # 4. Another Greedy with different shuffle
    random.seed(int(time.time() * 1000) % 10000)
    g2 = greedy_construct()
    inits.append(g2)
    
    # 5. Pseudo-random structured (e.g. multiples of small primes)
    # Try to hit many small differences
    # 1, 2, 4, 8, 16, 32, 64, 128, 256 ... powers of 2
    # But we need 32 elements.
    # Let's try a mix.
    mixed = list(range(1, 33))
    inits.append(mixed)

    # Phase 2: Polish each init with Greedy Local Search
    # This is the main improvement over pure SA.
    sa_time_per_run = 15
    polish_time_per_run = 5
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # First, polish
        polished, polished_score = greedy_polish(init, polish_time_per_run)
        
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall = polished
        
        # Then, SA on the polished result
        sa_limit = min(time.time() + sa_time_per_run, deadline - 10)
        s_res, score_res = simulated_annealing(polished, sa_limit)
        
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res

    # If we have time left, do one more long SA run on the best found
    if time.time() < deadline - 10 and best_overall is not None:
        remaining_time = deadline - 10 - time.time()
        s_res, score_res = simulated_annealing(best_overall, time.time() + remaining_time)
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res

    if best_overall is None:
        best_overall = list(range(1, N + 1))
        
    result = sorted(best_overall)
    # Final validation
    if len(set(result)) != N or not all(1 <= x <= RANGE for x in result):
        result = list(range(1, N + 1))
        
    return result
