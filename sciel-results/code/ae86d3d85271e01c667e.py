import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 32
    RANGE = 512

    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        distinct_d = set()
        # Optimization: iterate over pairs (x, z) with x < z
        # If (x+z) is even and (x+z)/2 is in s, then d = (z-x)/2 is valid.
        for i in range(len(s)):
            xi = s[i]
            for j in range(i + 1, len(s)):
                xj = s[j]
                if (xi + xj) % 2 == 0:
                    m = (xi + xj) // 2
                    if m in s_set:
                        distinct_d.add(m - xi)
        return len(distinct_d)

    def greedy_construct():
        """Construct a set by greedily adding elements that maximize distinct differences."""
        s = []
        s_set = set()
        
        for _ in range(N):
            best_val = -1
            best_score = -1
            
            # We can optimize this by only considering candidates that might add new diffs.
            # But 512 is small.
            # To break ties and add diversity, we can shuffle candidates.
            candidates = list(range(1, RANGE + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                if val in s_set:
                    continue
                
                # Temporarily add val
                new_s = s + [val]
                # We can compute score incrementally or just call score_set
                # score_set is fast enough for N=32.
                sc = score_set(new_s)
                
                if sc > best_score:
                    best_score = sc
                    best_val = val
            
            if best_val != -1:
                s.append(best_val)
                s_set.add(best_val)
            else:
                # Fallback
                s.append(random.randint(1, RANGE))
        
        return sorted(s)

    def simulated_annealing(s_init, time_limit):
        """SA with 1-replace and 2-swap moves."""
        s = list(s_init)
        current_score = score_set(s)
        best_s = list(s)
        best_score = current_score
        
        temp = 10.0
        cooling = 0.9995
        min_temp = 0.1
        
        while time.time() < time_limit and temp > min_temp:
            move_type = random.random()
            
            if move_type < 0.5:
                # 1-Replace Move
                i = random.randint(0, N - 1)
                old_val = s[i]
                new_val = random.randint(1, RANGE)
                
                if new_val == old_val:
                    continue
                if new_val in s:
                    continue
                
                s[i] = new_val
                new_score = score_set(s)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_val
            else:
                # 2-Swap Move
                i, j = random.sample(range(N), 2)
                old_i, old_j = s[i], s[j]
                
                new_i = random.randint(1, RANGE)
                new_j = random.randint(1, RANGE)
                
                if new_i == new_j:
                    continue
                if new_i in s and new_i != old_i and new_i != old_j:
                    continue
                if new_j in s and new_j != old_i and new_j != old_j:
                    continue
                
                s[i] = new_i
                s[j] = new_j
                new_score = score_set(s)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_i
                    s[j] = old_j
            
            temp *= cooling
            
        return best_s, best_score

    best_overall = None
    best_overall_score = 0
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # 1. Greedy
    g = greedy_construct()
    inits.append(g)
    
    # 2. Random
    inits.append(random.sample(range(1, RANGE + 1), N))
    
    # 3. Structured: Spread out elements
    # e.g., multiples of 16
    step = RANGE // N
    structured = [i * step + 1 for i in range(N)]
    inits.append(structured)
    
    # 4. Another Greedy with different shuffle
    random.seed(int(time.time() * 1000) % 10000) # Just to vary shuffle if needed, but random.shuffle uses global state
    g2 = greedy_construct()
    inits.append(g2)

    # Phase 2: SA on each init
    # Allocate time evenly, but stop if we find a very good solution
    sa_time_per_run = 20
    
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        sa_limit = min(time.time() + sa_time_per_run, deadline - 5)
        s_res, score_res = simulated_annealing(init, sa_limit)
        
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res
            
        # Early exit if we hit a theoretical high bound?
        # Max possible d is 255.
        if best_overall_score >= 150:
             # Continue for a bit more to see if we can push higher, but we have a good base.
             pass

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
