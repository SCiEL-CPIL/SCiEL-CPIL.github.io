import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    # Initialize with a few diverse starting points
    # 1. Small integers (1-20)
    # 2. Geometric progression (powers of 2 or 3, scaled to fit)
    # 3. Random small integers
    
    def gen_random_set():
        # Random distinct positive integers <= 10000
        # Bias towards smaller numbers as they tend to have denser sum/prod sets
        # but we want to balance. Let's just sample from 1..100 first, then maybe mix.
        # Actually, for SA, a good start is important.
        # Let's start with 1..20, which is a known decent baseline (score ~ 39 or 40?)
        # Wait, |1..20 + 1..20| = 39. |1..20 * 1..20| = 20*20 - overlaps.
        # Let's just use 1..20 as primary seed, and maybe a random one.
        return list(range(1, 21))
    
    def gen_geometric_set():
        # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 elems)
        # Need 20.
        # Let's try a mix of small smooth numbers.
        # This is just for diversity.
        elems = []
        for i in range(1, 11):
            elems.append(i)
        # Add some larger smooth numbers
        smooths = [12, 15, 18, 20, 24, 30, 36, 40, 48, 60]
        elems.extend(smooths)
        elems = sorted(list(set(elems)))
        while len(elems) < 20:
            elems.append(elems[-1] + 1)
        return elems[:20]

    seeds = [
        list(range(1, 21)),
        gen_geometric_set()
    ]
    
    best_set = None
    best_score = float('inf')
    
    # Candidate pool for swaps
    # We want to consider numbers that are likely to be useful.
    # Small numbers 1..100 are very likely candidates.
    # Also multiples of existing small numbers.
    # Let's define a static pool of "interesting" numbers.
    # 1..100 covers a lot.
    # Let's add some larger smooth numbers up to 1000.
    pool = set(range(1, 101))
    for i in range(1, 100):
        for j in range(1, 100):
            if i * j <= 1000:
                pool.add(i * j)
    # Add some random large numbers just in case? No, usually small/medium is better for minimizing set size.
    # Let's stick to 1..1000 for the pool.
    pool = list(pool)
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        A = seed[:]
        A = sorted(list(set(A)))
        # Ensure size 20
        if len(A) < 20:
            # Pad with small numbers not in A
            for i in range(1, 100):
                if i not in A:
                    A.append(i)
                    if len(A) == 20:
                        break
        if len(A) > 20:
            A = A[:20]
            
        current_score = calc_score(A)
        
        if current_score < best_score:
            best_score = current_score
            best_set = A[:]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        max_iter = 2000 # Adjust based on time
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Pick a random index to swap out
            idx = random.randint(0, len(A) - 1)
            old_val = A[idx]
            
            # Pick a new value from pool
            # To avoid picking the same value, we can pick from pool excluding A
            # But for speed, just pick random from pool. If it's in A, we might need to handle swap.
            # Simpler: Pick a random element from A to remove, and a random element from Pool to add.
            # If new_val is in A, we effectively just swapped indices, which is a no-op for the set.
            # We want to change the set.
            
            # Strategy: Pick a random x in A to remove. Pick a random y in Pool to add.
            # If y is in A, pick another y.
            # This is O(N) to check membership if we use a list, O(1) if set.
            # Let's maintain a set for fast lookup.
            
            A_set = set(A)
            
            # Remove old_val
            new_val = random.choice(pool)
            # Ensure new_val is not in A (unless it replaces itself, which is a no-op)
            # We want a distinct new set.
            # If new_val is in A, we didn't change the set.
            # Let's just pick a random candidate from Pool that is NOT in A.
            # Pre-computing "valid candidates" for each step is expensive.
            # Just try a few times.
            
            for _ in range(10):
                new_val = random.choice(pool)
                if new_val not in A_set:
                    break
            else:
                continue # If we couldn't find a new val, skip step
            
            # Perform swap
            A[idx] = new_val
            A.sort() # Keep sorted for determinism, though not strictly necessary for set calc
            A_set.remove(old_val)
            A_set.add(new_val)
            
            new_score = calc_score(A)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = A[:]
            else:
                # Reject, revert
                A[idx] = old_val
                A.sort()
                A_set.remove(new_val)
                A_set.add(old_val)
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
    return best_set

# Test run to ensure it works
if __name__ == "__main__":
    res = construct()
    print(res)
    print(calc_score(res))
