import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A. Uses symmetry to halve work."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def generate_smooth_pool(max_val=10000):
    """Generate a pool of smooth numbers (small prime factors) plus small integers."""
    pool = set()
    
    # Include all integers up to 200 for good sum structure
    for n in range(1, 201):
        pool.add(n)
    
    # Powers of 2, 3, 5, 7, 11, 13
    primes = [2, 3, 5, 7, 11, 13]
    
    def helper(idx, current):
        if current > max_val:
            return
        if current >= 1:
            pool.add(current)
        if idx < len(primes):
            p = primes[idx]
            while current * p <= max_val:
                current *= p
                helper(idx, current)
            helper(idx + 1, current)
    
    helper(0, 1)
    
    return sorted([x for x in pool if 1 <= x <= max_val])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 90s limit
    
    # Generate candidate pool
    candidate_pool = generate_smooth_pool(10000)
    pool_size = len(candidate_pool)
    
    # Initial strategies
    initial_sets = []
    
    # 1. Pure geometric: powers of 2
    gp2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in gp2):
        initial_sets.append(gp2)
    
    # 2. Powers of 3
    gp3 = [3**i for i in range(20)]
    if all(x <= 10000 for x in gp3):
        initial_sets.append(gp3)
    
    # 3. Mixed smooth numbers in rough geometric progression
    log_targets = [i * 0.15 for i in range(20)]
    selected = []
    for target in log_targets:
        target_val = math.exp(target * math.log(2))
        best = min(candidate_pool, key=lambda x: abs(math.log(x) - target))
        if best not in selected:
            selected.append(best)
    if len(selected) == 20:
        initial_sets.append(sorted(selected))
    
    # 4. Arithmetic progression (good for sums)
    initial_sets.append(list(range(1, 21)))
    
    # 5. Geometric with ratio 1.5
    gp15 = []
    val = 1.0
    for i in range(20):
        gp15.append(round(val))
        val *= 1.5
    gp15 = list(set(gp15))
    if len(gp15) >= 20:
        initial_sets.append(sorted(gp15[:20]))
    
    # 6. Powers of 2 up to 2^9, then powers of 3
    mixed1 = [2**i for i in range(10)] + [3**i for i in range(1, 11)]
    mixed1 = sorted(set(mixed1))
    if len(mixed1) >= 20:
        initial_sets.append(mixed1[:20])
    
    # 7. Small numbers + powers of 2
    mixed2 = list(range(1, 6)) + [2**i for i in range(2, 14)]
    mixed2 = sorted(set(mixed2))
    if len(mixed2) >= 20:
        initial_sets.append(mixed2[:20])
    
    # 8. First 20 smooth numbers
    initial_sets.append(candidate_pool[:20])
    
    # 9. Squares
    squares = [i*i for i in range(1, 21)]
    if all(x <= 10000 for x in squares):
        initial_sets.append(squares)
    
    # 10. Odd numbers
    initial_sets.append([1 + 2*i for i in range(20)])
    
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
    
    # Simulated annealing with targeted moves
    T = 5.0
    T_min = 0.1
    cooling = 0.9997
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if time.time() >= deadline:
            break
        
        # Choose move type: single replacement (80%), swap (15%), double replacement (5%)
        move_type = random.random()
        
        if move_type < 0.80:
            # Single replacement
            idx = random.randint(0, 19)
            
            # Choose candidate: 50% from near old value, 50% random from pool
            old_val = current_A[idx]
            if random.random() < 0.5:
                # Near old value
                candidates = [c for c in candidate_pool if old_val // 2 <= c <= old_val * 2 and c != old_val]
                if candidates:
                    new_val = random.choice(candidates)
                else:
                    new_val = candidate_pool[random.randint(0, pool_size - 1)]
            else:
                new_val = candidate_pool[random.randint(0, pool_size - 1)]
            
            if new_val in current_A:
                T = max(T * cooling, T_min)
                continue
            
            trial_A = current_A[:]
            trial_A[idx] = new_val
            trial_A.sort()
            
            if len(set(trial_A)) != 20:
                T = max(T * cooling, T_min)
                continue
            
            trial_score, _, _ = compute_metrics(trial_A)
            
        elif move_type < 0.95:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            trial_A = current_A[:]
            trial_A[i], trial_A[j] = trial_A[j], trial_A[i]
            
            trial_score, _, _ = compute_metrics(trial_A)
            
        else:
            # Double replacement
            i, j = random.sample(range(20), 2)
            new_i = candidate_pool[random.randint(0, pool_size - 1)]
            new_j = candidate_pool[random.randint(0, pool_size - 1)]
            
            if new_i in current_A or new_j in current_A or new_i == new_j:
                T = max(T * cooling, T_min)
                continue
            
            trial_A = current_A[:]
            trial_A[i] = new_i
            trial_A[j] = new_j
            trial_A.sort()
            
            if len(set(trial_A)) != 20:
                T = max(T * cooling, T_min)
                continue
            
            trial_score, _, _ = compute_metrics(trial_A)
        
        delta = trial_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_A = trial_A[:]
            current_score = trial_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            no_improve_count += 1
        
        T = max(T * cooling, T_min)
        
        # If stuck, perturb
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Perturb current solution
            for _ in range(3):
                idx = random.randint(0, 19)
                new_val = candidate_pool[random.randint(0, pool_size - 1)]
                if new_val not in current_A:
                    current_A[idx] = new_val
                    current_A.sort()
                    if len(set(current_A)) == 20:
                        current_score, _, _ = compute_metrics(current_A)
                        break
    
    return best_A
