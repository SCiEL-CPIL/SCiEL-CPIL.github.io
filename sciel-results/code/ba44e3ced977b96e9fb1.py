import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # Initial set
    U = [0, 1, 3]
    
    def get_score(u):
        if not u:
            return 0.0
        max_u = max(u)
        if max_u == 0:
            return 0.0
        # Calculate sumset and difference set
        # Using sets for uniqueness
        # For performance, if len(u) is large, this might be slow.
        # Let's cap the size or time.
        u_set = set(u)
        sum_set = set()
        diff_set = set()
        # Optimization: if len(u) > 500, this might take a while.
        # 500^2 = 250,000 ops. Should be < 0.1s in Python? Maybe 0.5s.
        # Let's be careful.
        for a in u_set:
            for b in u_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        if not sum_set or not diff_set:
            return 0.0
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom

    current_score = get_score(U)
    best_U = U[:]
    best_score = current_score
    
    max_val = 1000  # Search space for new elements
    
    # Local search
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        # Sample a few candidates to avoid full scan if max_val is large
        # But max_val is 1000, so full scan is 1000 * (cost of eval)
        # Cost of eval is ~ O(N^2). If N=100, 10k ops. 1000*10k = 10M ops.
        # This might be too slow per iteration if we do it for every x.
        # We should only evaluate if x is "promising" or use a faster heuristic.
        
        # Better: Only evaluate a subset of candidates per iteration.
        candidates = list(range(0, max_val + 1))
        random.shuffle(candidates)
        
        # Try a limited number of additions
        for x in candidates[:20]:
            if x in U:
                continue
            new_U = U + [x]
            # Heuristic check: if x is too large, score might drop due to denominator
            # But let's just compute.
            s = get_score(new_U)
            if s > current_score:
                U = new_U
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_U = U[:]
                improved = True
                no_improvement_count = 0
                break # Restart candidate search with new U
        
        # Try removing elements
        if not improved and len(U) > 2:
            removal_candidates = [x for x in U if x != 0]
            random.shuffle(removal_candidates)
            for x in removal_candidates[:10]:
                new_U = [y for y in U if y != x]
                s = get_score(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = U[:]
                    improved = True
                    no_improvement_count = 0
                    break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Random perturbation: Add a random element
                x = random.randint(1, max_val)
                if x not in U:
                    U.append(x)
                    U.sort()
                    current_score = get_score(U)
                no_improvement_count = 0
                max_no_improvement += 10 # Increase patience slightly
        
        # Increase max_val slowly to allow larger sets
        if len(U) < 100:
             max_val = min(max_val + 10, 5000)
        elif len(U) < 500:
             max_val = min(max_val + 5, 20000)
        else:
             max_val = min(max_val + 1, 50000)

    return best_U
