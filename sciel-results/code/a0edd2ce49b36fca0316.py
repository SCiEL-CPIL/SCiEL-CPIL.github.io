import time
import random
import itertools

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    sum_set = set()
    prod_set = set()
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Initial set: powers of 2 (geometric progression minimizes product set)
    # But we need to balance sum and product sets
    # Start with a mix: small numbers for sums, structured for products
    
    # Strategy: Use numbers with many factors (smooth numbers) and geometric structure
    # Try several initial configurations and pick the best
    
    initial_sets = [
        # Geometric progression with ratio 2
        [2**i for i in range(20)],
        # Mixed: powers of 2 and 3
        sorted(set([2**i for i in range(10)] + [3**i for i in range(10)])),
        # Smooth numbers up to some bound
        sorted([n for n in range(1, 10001) if all(n % p != 0 or n // p**2 >= 0 for p in [2,3,5,7,11,13,17,19,23,29,31])][:20]),
        # Consecutive integers (baseline)
        list(range(1, 21)),
        # Powers of 2 with some variation
        [2**i for i in range(15)] + [3**i for i in range(5)],
    ]
    
    # Filter to valid sets (distinct, positive, <= 10000, exactly 20)
    valid_initials = []
    for s in initial_sets:
        s = list(set(s))
        s = [x for x in s if 1 <= x <= 10000]
        if len(s) >= 20:
            s = s[:20]
        if len(s) == 20:
            valid_initials.append(s)
    
    if not valid_initials:
        valid_initials = [list(range(1, 21))]
    
    # Pick the best initial set
    best_score = float('inf')
    best_A = valid_initials[0][:]
    for s in valid_initials:
        score, _, _ = compute_metrics(s)
        if score < best_score:
            best_score = score
            best_A = s[:]
    
    # Local search: try replacing elements with better candidates
    # Candidate pool: smooth numbers and numbers with many divisors
    candidate_pool = set()
    for n in range(1, 10001):
        # Count divisors
        div_count = sum(1 for d in range(1, int(n**0.5) + 1) if n % d == 0)
        if div_count >= 4:  # Numbers with many divisors
            candidate_pool.add(n)
        # Also include powers of small primes
        for p in [2, 3, 5, 7, 11, 13]:
            power = 1
            while power <= 10000:
                candidate_pool.add(power)
                power *= p
    
    candidate_pool = list(candidate_pool)
    random.seed(42)
    
    # Local search iterations
    current_A = best_A[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Try random replacements
        improved = False
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Pick a random candidate to replace with
            new_val = random.choice(candidate_pool)
            
            if new_val in current_A:
                continue
            
            # Try the replacement
            trial_A = current_A[:]
            trial_A[idx] = new_val
            trial_A.sort()
            
            # Ensure distinctness and validity
            if len(set(trial_A)) != 20 or any(x < 1 or x > 10000 for x in trial_A):
                continue
            
            trial_score, _, _ = compute_metrics(trial_A)
            
            if trial_score < current_score:
                current_A = trial_A
                current_score = trial_score
                improved = True
        
        # If no improvement in this batch, try a different strategy
        if not improved:
            # Try hill climbing with multiple swaps
            for _ in range(50):
                if time.time() >= deadline:
                    break
                
                # Pick two random elements to swap
                i, j = random.sample(range(20), 2)
                trial_A = current_A[:]
                trial_A[i], trial_A[j] = trial_A[j], trial_A[i]
                
                trial_score, _, _ = compute_metrics(trial_A)
                
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
        
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
    
    return best_A
