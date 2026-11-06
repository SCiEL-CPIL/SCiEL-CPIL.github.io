import time
import random

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    sum_set = set()
    prod_set = set()
    for i, a in enumerate(A):
        for j, b in enumerate(A):
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 90s limit
    
    # Generate smooth number candidate pool (numbers with only small prime factors)
    # These create more product collisions
    candidate_pool = set()
    primes = [2, 3, 5, 7, 11, 13]
    
    def generate_smooth(max_val, primes):
        result = set()
        def helper(idx, current):
            if current > max_val:
                return
            if current >= 1:
                result.add(current)
            if idx < len(primes):
                p = primes[idx]
                while current * p <= max_val:
                    current *= p
                    helper(idx, current)
                helper(idx + 1, current)
        helper(0, 1)
        return result
    
    smooth_numbers = generate_smooth(10000, primes)
    candidate_pool = sorted(smooth_numbers)
    
    # Initial strategies: geometric progressions with different bases
    initial_sets = []
    
    # Pure geometric: powers of 2
    gp2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in gp2):
        initial_sets.append(gp2)
    
    # Powers of 3
    gp3 = [3**i for i in range(20)]
    if all(x <= 10000 for x in gp3[:20]):
        initial_sets.append(gp3[:20])
    
    # Mixed smooth numbers: pick 20 smooth numbers that form a rough geometric progression
    # Select smooth numbers that are spread out multiplicatively
    log_targets = [i * 0.15 for i in range(20)]  # Target log values
    selected = []
    for target in log_targets:
        import math
        target_val = math.exp(target * math.log(2))
        # Find closest smooth number
        best = min(candidate_pool, key=lambda x: abs(math.log(x) - target))
        if best not in selected:
            selected.append(best)
    if len(selected) == 20:
        initial_sets.append(sorted(selected))
    
    # Arithmetic progression (good for sums)
    initial_sets.append(list(range(1, 21)))
    
    # Geometric with ratio 1.5
    gp15 = []
    val = 1
    for i in range(20):
        gp15.append(round(val))
        val *= 1.5
    gp15 = list(set(gp15))
    if len(gp15) >= 20:
        initial_sets.append(sorted(gp15[:20]))
    
    # Filter valid initial sets
    valid_initials = []
    for s in initial_sets:
        s = list(set(s))
        s = [x for x in s if 1 <= x <= 10000]
        if len(s) >= 20:
            s = sorted(s[:20])
        if len(s) == 20:
            valid_initials.append(s)
    
    if not valid_initials:
        valid_initials = [list(range(1, 21))]
    
    # Pick best initial set
    best_score = float('inf')
    best_A = valid_initials[0][:]
    for s in valid_initials:
        score, _, _ = compute_metrics(s)
        if score < best_score:
            best_score = score
            best_A = s[:]
    
    random.seed(42)
    current_A = best_A[:]
    current_score = best_score
    
    # Local search with targeted replacements
    # Strategy: for each position, try replacing with smooth numbers that
    # create collisions in both sum and product sets
    
    # Precompute current sum and product sets for incremental updates
    while time.time() < deadline:
        improved = False
        
        # Try single replacements with smooth candidates
        for idx in range(20):
            if time.time() >= deadline:
                break
            
            old_val = current_A[idx]
            # Try a subset of candidates near the old value's "position"
            # and some random smooth numbers
            candidates_to_try = []
            
            # Smooth numbers in range [old_val//3, old_val*3]
            for c in candidate_pool:
                if old_val // 3 <= c <= old_val * 3 and c != old_val:
                    candidates_to_try.append(c)
            
            # Add some random smooth numbers
            if len(candidate_pool) > 20:
                random_sample = random.sample(candidate_pool, min(10, len(candidate_pool)))
                candidates_to_try.extend(random_sample)
            
            for new_val in candidates_to_try[:30]:  # Limit per position
                if new_val in current_A:
                    continue
                
                trial_A = current_A[:]
                trial_A[idx] = new_val
                trial_A.sort()
                
                if len(set(trial_A)) != 20:
                    continue
                
                trial_score, _, _ = compute_metrics(trial_A)
                
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
                    break  # Move to next position after improvement
        
        # If no improvement, try pairwise swaps
        if not improved:
            for _ in range(100):
                if time.time() >= deadline:
                    break
                
                i, j = random.sample(range(20), 2)
                # Try replacing both with new smooth values
                trial_A = current_A[:]
                
                # Try swapping
                trial_A[i], trial_A[j] = trial_A[j], trial_A[i]
                trial_score, _, _ = compute_metrics(trial_A)
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
                    break
                
                # Try replacing both with new values
                new_i = random.choice(candidate_pool)
                new_j = random.choice(candidate_pool)
                if new_i not in current_A and new_j not in current_A and new_i != new_j:
                    trial_A2 = current_A[:]
                    trial_A2[i] = new_i
                    trial_A2[j] = new_j
                    trial_A2.sort()
                    if len(set(trial_A2)) == 20:
                        trial_score2, _, _ = compute_metrics(trial_A2)
                        if trial_score2 < current_score:
                            current_A = trial_A2
                            current_score = trial_score2
                            improved = True
                            break
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
        
        # If stuck for too long, restart from a different initial
        if not improved and random.random() < 0.1:
            # Perturb current solution
            for _ in range(5):
                idx = random.randint(0, 19)
                new_val = random.choice(candidate_pool)
                if new_val not in current_A:
                    current_A[idx] = new_val
                    current_A.sort()
                    if len(set(current_A)) == 20:
                        current_score, _, _ = compute_metrics(current_A)
                        break
    
    return best_A
