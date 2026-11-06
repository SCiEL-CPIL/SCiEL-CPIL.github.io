import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        diffs = set()
        for a in s:
            # a + 2*d <= 512 => d <= (512 - a) // 2
            max_d = (512 - a) // 2
            for d in range(1, max_d + 1):
                if a + d in s_set and a + 2 * d in s_set:
                    diffs.add(d)
        return len(diffs)
    
    def fast_score(s_set, s_list):
        """Faster scoring using set membership."""
        diffs = set()
        for a in s_list:
            max_d = (512 - a) // 2
            for d in range(1, max_d + 1):
                if (a + d) in s_set and (a + 2 * d) in s_set:
                    diffs.add(d)
        return len(diffs)
    
    def local_search(s, max_iters=100000):
        """Aggressive local search: try all single-element swaps."""
        s = list(s)
        s_set = set(s)
        current_score = fast_score(s_set, s)
        best_score = current_score
        best_s = list(s)
        
        n = len(s)
        no_improve = 0
        max_no_improve = 2000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping each element with each possible new value
            for i in range(n):
                old = s[i]
                # Try a subset of candidate replacements
                # Use deterministic candidates: nearby values and random
                candidates = set()
                # Nearby values
                for delta in range(-20, 21):
                    v = old + delta
                    if 1 <= v <= 512 and v != old:
                        candidates.add(v)
                # Some random
                for _ in range(30):
                    v = random.randint(1, 512)
                    if v != old:
                        candidates.add(v)
                
                for new_val in candidates:
                    if new_val in s_set:
                        continue
                    
                    s[i] = new_val
                    s_set.remove(old)
                    s_set.add(new_val)
                    
                    new_score = fast_score(s_set, s)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_s = list(s)
                        no_improve = 0
                        break
                    else:
                        s[i] = old
                        s_set.remove(new_val)
                        s_set.add(old)
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
                # Perturb: randomly change 2 elements
                for _ in range(2):
                    i = random.randint(0, n - 1)
                    old = s[i]
                    new_val = random.randint(1, 512)
                    while new_val in s_set:
                        new_val = random.randint(1, 512)
                    s[i] = new_val
                    s_set.remove(old)
                    s_set.add(new_val)
                current_score = fast_score(s_set, s)
        
        return best_s, best_score
    
    def simulated_annealing(s, max_iters=50000):
        """Simulated annealing."""
        s = list(s)
        s_set = set(s)
        current_score = fast_score(s_set, s)
        best_score = current_score
        best_s = list(s)
        
        temp = 10.0
        cooling = 0.99995
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, len(s) - 1)
            old_val = s[idx]
            new_val = random.randint(1, 512)
            
            if new_val in s_set:
                continue
            
            s[idx] = new_val
            s_set.remove(old_val)
            s_set.add(new_val)
            
            new_score = fast_score(s_set, s)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_s = list(s)
            else:
                s[idx] = old_val
                s_set.remove(new_val)
                s_set.add(old_val)
            
            temp *= cooling
        
        return best_s, best_score
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. Consecutive range [1, 32]
    candidates.append(list(range(1, 33)))
    
    # 2. Consecutive range starting at different points
    for start in [1, 10, 20, 50, 100, 200]:
        if start + 31 <= 512:
            candidates.append(list(range(start, start + 32)))
    
    # 3. Two clusters
    for gap in [20, 40, 60, 100, 150]:
        c1 = list(range(1, 17))  # 16 elements
        c2 = list(range(1 + gap, 1 + gap + 16))  # 16 elements
        if max(c2) <= 512:
            candidates.append(c1 + c2)
    
    # 4. Arithmetic progression with small step
    for step in [1, 2, 3]:
        ap = list(range(1, 1 + 32 * step, step))
        if len(ap) == 32 and max(ap) <= 512:
            candidates.append(ap)
    
    # 5. Random sets
    for _ in range(10):
        candidates.append(random.sample(range(1, 513), 32))
    
    # 6. Structured: powers and multiples
    structured = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    while len(structured) < 32:
        v = random.randint(1, 512)
        if v not in structured:
            structured.append(v)
    candidates.append(structured)
    
    # 7. Dense in middle
    mid = list(range(200, 232))
    candidates.append(mid)
    
    # 8. Sparse but spread
    spread = [int(1 + i * 511 / 31) for i in range(32)]
    # Ensure distinct
    seen = set()
    unique_spread = []
    for v in spread:
        v = min(max(v, 1), 512)
        while v in seen:
            v = min(v + 1, 512)
        if v <= 512:
            seen.add(v)
            unique_spread.append(v)
    while len(unique_spread) < 32:
        v = random.randint(1, 512)
        if v not in seen:
            seen.add(v)
            unique_spread.append(v)
    candidates.append(unique_spread)
    
    best_overall = None
    best_overall_score = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Local search
        ls_result, ls_score = local_search(cand, max_iters=30000)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall = ls_result
        
        # Simulated annealing from the locally searched result
        if time.time() < deadline:
            sa_result, sa_score = simulated_annealing(ls_result, max_iters=20000)
            if sa_score > best_overall_score:
                best_overall_score = sa_score
                best_overall = sa_result
    
    # Final local search on the best
    if best_overall is not None and time.time() < deadline:
        final, final_score = local_search(best_overall, max_iters=50000)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final
    
    if best_overall is None:
        best_overall = list(range(1, 33))
    
    return best_overall
