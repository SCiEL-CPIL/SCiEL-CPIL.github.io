import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)
    
    def score(A):
        # A is a tuple or list of 20 distinct ints
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        return max(len(sum_set), len(prod_set))
    
    def generate_ap_seeds():
        """Generate AP-based seeds that minimize sumset size."""
        candidates = []
        # Standard AP: 1, 2, ..., 20
        candidates.append(tuple(range(1, 21)))
        # Scaled AP: 2, 4, 6, ..., 40
        candidates.append(tuple(2 * i for i in range(1, 21)))
        # AP with larger step: 10, 20, ..., 200
        candidates.append(tuple(10 * i for i in range(1, 21)))
        # AP starting at 1 with step 3: 1, 4, 7, ..., 58
        candidates.append(tuple(1 + 3 * i for i in range(20)))
        return candidates
    
    def generate_gp_seeds():
        """Generate GP-based seeds for product-set collisions."""
        candidates = []
        # Pure powers of 2 (but only 14 fit in 10000)
        # So we need to be creative
        # Powers of 2 up to 2^13 = 8192, then add some perturbed values
        base_gp = tuple(2 ** i for i in range(14))  # 1, 2, 4, ..., 8192
        # Extend with 6 more values that are products of small primes
        extra = [3, 5, 7, 9, 11, 13]
        candidates.append(base_gp + tuple(extra))
        
        # Smaller base: 3^0 to 3^8 = 1 to 6561, then extras
        gp3 = tuple(3 ** i for i in range(9))  # 1, 3, 9, ..., 6561
        extra3 = [2, 4, 5, 7, 8, 10, 12, 14]
        candidates.append(gp3 + tuple(extra3))
        
        # Mixed: powers of 2 and 3
        mixed = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96]
        candidates.append(tuple(mixed))
        
        return candidates
    
    def generate_smooth_seeds():
        """Generate sets of smooth numbers for both sum and product collisions."""
        candidates = []
        # 5-smooth numbers (Hamming numbers)
        smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
        candidates.append(tuple(smooth5))
        
        # 7-smooth numbers
        smooth7 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27]
        candidates.append(tuple(smooth7))
        
        # Powers of 2 with small multipliers
        pow2_mult = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96]
        candidates.append(tuple(pow2_mult))
        
        return candidates
    
    def generate_random_seeds(n_seeds=10):
        """Generate random seeds."""
        candidates = []
        for _ in range(n_seeds):
            candidates.append(tuple(random.sample(range(1, 10001), 20)))
        return candidates
    
    def local_search(A, score_fn, max_iterations=5000, time_limit=10.0):
        """Perform local search with multiple move types."""
        current = list(A)
        current_score = score_fn(tuple(current))
        best = list(current)
        best_score = current_score
        start = time.time()
        
        for iteration in range(max_iterations):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, 19)
            old_val = current[idx]
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Small offset: ±1 to ±5
                new_val = old_val + random.randint(-5, 5)
            elif move_type < 0.5:
                # Multiplicative perturbation: multiply or divide by small factor
                factor = random.choice([2, 3, 4, 5, 6])
                if random.random() < 0.5:
                    new_val = old_val * factor
                else:
                    new_val = old_val // factor if old_val % factor == 0 else old_val // factor + 1
            elif move_type < 0.7:
                # Swap with another element
                jdx = random.randint(0, 19)
                if jdx == idx:
                    continue
                new_val = current[jdx]
                # Swap
                current[idx], current[jdx] = current[jdx], current[idx]
                new_A = tuple(current)
                new_score = score_fn(new_A)
                if new_score < best_score:
                    best = list(current)
                    best_score = new_score
                    current_score = new_score
                elif new_score < current_score:
                    current_score = new_score
                else:
                    # Revert swap
                    current[idx], current[jdx] = current[jdx], current[idx]
                continue
            elif move_type < 0.85:
                # Random small value [1, 50]
                new_val = random.randint(1, 50)
            else:
                # Random in [1, 10000]
                new_val = random.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000 or new_val == old_val:
                continue
            if new_val in current:
                continue
            
            new_A_list = list(current)
            new_A_list[idx] = new_val
            new_A = tuple(new_A_list)
            new_score = score_fn(new_A)
            
            if new_score < current_score:
                current = new_A_list
                current_score = new_score
                if current_score < best_score:
                    best = list(current)
                    best_score = current_score
            else:
                # Accept with some probability (simulated annealing style)
                delta = new_score - current_score
                if delta <= 3 and random.random() < 0.3:
                    current = new_A_list
                    current_score = new_score
        
        return tuple(best), best_score
    
    # Generate all candidate seeds
    all_candidates = []
    all_candidates.extend(generate_ap_seeds())
    all_candidates.extend(generate_gp_seeds())
    all_candidates.extend(generate_smooth_seeds())
    all_candidates.extend(generate_random_seeds(5))
    
    # Evaluate all candidates and pick the best few for local search
    scored_candidates = []
    for c in all_candidates:
        if len(set(c)) != 20:
            continue
        if max(c) > 10000:
            continue
        s = score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 candidates for local search
    best_A = None
    best_score = float('inf')
    
    top_k = min(5, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 15:
            break
        s, c = scored_candidates[i]
        if s >= best_score:
            continue
        
        # Run local search on this candidate
        time_remaining = deadline - time.time() - 5
        if time_remaining < 5:
            break
        
        # Allocate time based on rank
        if i == 0:
            search_time = min(30, time_remaining * 0.5)
        elif i == 1:
            search_time = min(20, time_remaining * 0.3)
        else:
            search_time = min(10, time_remaining * 0.2)
        
        result_A, result_score = local_search(c, score, time_limit=search_time)
        
        if result_score < best_score:
            best_score = result_score
            best_A = result_A
    
    # Final polish: try to improve the best solution
    if best_A is not None and time.time() < deadline - 3:
        polish_A, polish_score = local_search(best_A, score, max_iterations=2000, time_limit=min(5, deadline - time.time() - 2))
        if polish_score < best_score:
            best_score = polish_score
            best_A = polish_A
    
    if best_A is None:
        # Fallback: return AP
        best_A = tuple(range(1, 21))
    
    return list(best_A)
