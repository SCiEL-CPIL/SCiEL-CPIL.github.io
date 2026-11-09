import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all sums (2 to 20000)
    # Using a sieve-like approach for speed
    sum_primes = [set() for _ in range(2 * max_val + 1)]
    for p in range(2, 2 * max_val + 1):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, 2 * max_val + 1, p):
                sum_primes[multiple].add(p)
                
    def get_initial_set(strategy):
        if strategy == 0:
            return random.sample(range(1, max_val + 1), target_size)
        elif strategy == 1:
            return list(range(1, target_size + 1))
        elif strategy == 2:
            return [2 * i for i in range(1, target_size + 1)]
        elif strategy == 3:
            return [3 * i for i in range(1, target_size + 1)]
        else:
            # Random smooth-ish numbers
            candidates = [i for i in range(1, max_val + 1) if len(sum_primes[i]) <= 3]
            if len(candidates) < target_size:
                return random.sample(range(1, max_val + 1), target_size)
            return random.sample(candidates, target_size)

    def calculate_state(A):
        counts = {}
        active = set()
        for i in range(target_size):
            for j in range(i + 1, target_size):
                s = A[i] + A[j]
                for p in sum_primes[s]:
                    if p not in counts:
                        counts[p] = 0
                        active.add(p)
                    counts[p] += 1
        return counts, active

    best_set = None
    best_score = float('inf')
    
    strategies = [0, 1, 2, 3, 4]
    strategy_idx = 0
    
    while time.time() < deadline:
        # Initialize
        A = get_initial_set(strategies[strategy_idx % len(strategies)])
        strategy_idx += 1
        
        counts, active = calculate_state(A)
        current_score = len(active)
        
        if current_score < best_score:
            best_score = current_score
            best_set = A[:]
            if best_score <= 4:
                break # Very good, likely optimal or near-optimal
                
        no_improve_steps = 0
        max_no_improve = 2000 + random.randint(0, 1000)
        
        # Local Search
        while time.time() < deadline - 2.0:
            if no_improve_steps > max_no_improve:
                break
                
            i = random.randint(0, target_size - 1)
            x = random.randint(1, max_val)
            
            # Ensure distinctness
            if x == A[i]:
                continue
            if x in A:
                continue
                
            # Calculate delta
            old_score = current_score
            new_counts = counts.copy()
            new_active = active.copy()
            
            # Remove contributions of A[i]
            for j in range(target_size):
                if i == j: continue
                s_old = A[i] + A[j]
                for p in sum_primes[s_old]:
                    new_counts[p] -= 1
                    if new_counts[p] == 0:
                        new_active.discard(p)
                        
            # Add contributions of x
            for j in range(target_size):
                if i == j: continue
                s_new = x + A[j]
                for p in sum_primes[s_new]:
                    if p not in new_counts:
                        new_counts[p] = 0
                        new_active.add(p)
                    new_counts[p] += 1
                    
            new_score = len(new_active)
            
            if new_score < old_score:
                # Accept
                A[i] = x
                counts = new_counts
                active = new_active
                current_score = new_score
                no_improve_steps = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_set = A[:]
                    if best_score <= 4:
                        break
            else:
                # Reject or accept with small probability to escape local minima
                # If scores are equal, accept to explore
                if new_score == old_score:
                    if random.random() < 0.5:
                        A[i] = x
                        counts = new_counts
                        active = new_active
                        current_score = new_score
                else:
                    no_improve_steps += 1
                    
        if best_score <= 4:
            break

    return best_set

# Fallback if construct fails or returns None (should not happen)
def construct_fallback():
    return list(range(1, 13))

# Ensure construct is the entry point
# The task says "Keep the function names and signatures of the seed program"
# The seed program has `construct()`.
