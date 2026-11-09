import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        a = np.array(A, dtype=np.int64)
        # Vectorized sum and product sets
        sumset = a[:, None] + a[None, :]
        prodset = a[:, None] * a[None, :]
        
        # Use np.unique which is efficient for small arrays
        s_size = len(np.unique(sumset))
        p_size = len(np.unique(prodset))
        return max(s_size, p_size)

    def get_grid_set():
        """Generate a 4x5 grid of 2^i * 3^j."""
        vals = []
        for i in range(4):
            for j in range(5):
                v = (2 ** i) * (3 ** j)
                if v <= max_val:
                    vals.append(v)
        vals = sorted(set(vals))
        if len(vals) < n:
            # Fill with next smooth numbers if needed
            # But 4*5=20, max is 8*81=648, so it's fine.
            pass
        return vals[:n]

    def get_smooth_numbers(k, limit, count):
        """Generate smallest 'count' k-smooth numbers <= limit."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        primes = [p for p in primes if p <= k]
        
        smooth_list = []
        for x in range(1, limit + 1):
            t = x
            for p in primes:
                while t % p == 0:
                    t //= p
            if t == 1:
                smooth_list.append(x)
                if len(smooth_list) == count:
                    return smooth_list
        return smooth_list

    def local_search(start_set, max_iter=50000):
        current = start_set[:]
        current_score = evaluate(current)
        local_best = current[:]
        local_best_score = current_score
        
        # Simulated Annealing parameters
        T = 10.0
        cooling = 0.9995
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # Generate candidate new_val
            r = random.random()
            if r < 0.2:
                # Random small integer
                new_val = random.randint(1, 100)
            elif r < 0.4:
                # Random in range
                new_val = random.randint(1, max_val)
            elif r < 0.7:
                # Smooth number near old_val
                # Try multiplying/dividing by small primes
                base = old_val
                mult = random.choice([2, 3, 5, 7, 1/2, 1/3, 1/5])
                new_val = int(base * mult)
                if new_val < 1: new_val = 1
                if new_val > max_val: new_val = max_val
            else:
                # Grid-like move: replace with 2^i 3^j
                i_exp = random.randint(0, 5)
                j_exp = random.randint(0, 5)
                new_val = (2 ** i_exp) * (3 ** j_exp)
                if new_val > max_val:
                    new_val = random.randint(1, max_val)

            if new_val in current:
                continue
                
            # Create new set
            temp_set = current[:]
            temp_set[idx] = new_val
            temp_set.sort()
            
            # Evaluate
            new_score = evaluate(temp_set)
            
            # Acceptance criterion
            if new_score < current_score:
                current = temp_set
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best = current[:]
            else:
                # Accept worse moves with probability exp(-delta/T)
                delta = new_score - current_score
                if delta < 0: # Should not happen if we checked <
                    pass
                prob = np.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    current = temp_set
                    current_score = new_score
            
            T *= cooling

        return local_best, local_best_score

    # Initial Candidates
    candidates = []
    
    # 1. Multiplicative Grid (2^i * 3^j)
    c_grid = get_grid_set()
    if len(c_grid) == n:
        candidates.append(c_grid)
        
    # 2. Smallest 3-smooth
    c_3smooth = get_smooth_numbers(3, max_val, n)
    if len(c_3smooth) == n:
        candidates.append(c_3smooth)
        
    # 3. Smallest 5-smooth
    c_5smooth = get_smooth_numbers(5, max_val, n)
    if len(c_5smooth) == n:
        candidates.append(c_5smooth)
        
    # 4. Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # 5. Random
    for _ in range(3):
        cand = random.sample(range(1, max_val + 1), n)
        candidates.append(sorted(cand))

    # Run local search
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Validate
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            continue
            
        res_set, res_score = local_search(cand)
        
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
