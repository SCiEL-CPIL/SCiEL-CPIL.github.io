import random
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Bit widths for each prime's exponent in the packed representation
    bit_widths = [3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    shifts = [0] * num_primes
    for i in range(1, num_primes):
        shifts[i] = shifts[i-1] + bit_widths[i-1]
        
    # Precompute packed vectors
    packed_vecs = {}
    for n in range(2, 41):
        p_val = 0
        temp = n
        for i, p in enumerate(primes):
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            p_val |= (exp << shifts[i])
        packed_vecs[n] = p_val
        
    candidates = list(range(2, 41))
    num_candidates = len(candidates)
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Initial set: primes
    initial_set = primes[:]
    
    def get_sums(current_set):
        sums = {0}
        for n in current_set:
            v = packed_vecs[n]
            new_sums = {s + v for s in sums}
            if new_sums & sums:
                return None
            sums = new_sums
        return sums
        
    def is_valid(current_set):
        return get_sums(current_set) is not None
        
    best_set = initial_set[:]
    best_len = len(initial_set)
    
    start_time = time.time()
    deadline = start_time + 55
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random restart
        order = list(range(num_candidates))
        rng.shuffle(order)
        
        current_set = []
        sums = {0}
        in_set = [False] * num_candidates
        
        for idx in order:
            n = candidates[idx]
            v = packed_vecs[n]
            new_sums = {s + v for s in sums}
            if not (new_sums & sums):
                sums = new_sums
                current_set.append(n)
                in_set[idx] = True
        
        current_len = len(current_set)
        
        if current_len > best_len:
            best_len = current_len
            best_set = current_set[:]
            if best_len >= 18:
                break
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try 1-swaps
            for i in range(len(current_set)):
                if time.time() > deadline: break
                n_remove = current_set[i]
                remove_idx = val_to_idx[n_remove]
                
                for j in range(num_candidates):
                    if in_set[j]: continue
                    n_add = candidates[j]
                    
                    new_set = current_set[:]
                    new_set[i] = n_add
                    if is_valid(new_set):
                        current_set = new_set
                        in_set[remove_idx] = False
                        in_set[j] = True
                        improved = True
                        break
                if improved: break
            
            if improved or time.time() > deadline: continue
            
            # Try 2-swaps
            if len(current_set) >= 2:
                # Limit attempts to avoid slowing down
                for _ in range(20):
                    if time.time() > deadline: break
                    i1, i2 = rng.sample(range(len(current_set)), 2)
                    n1, n2 = current_set[i1], current_set[i2]
                    idx1, idx2 = val_to_idx[n1], val_to_idx[n2]
                    
                    available = [j for j in range(num_candidates) if not in_set[j]]
                    if len(available) < 2: continue
                    j1, j2 = rng.sample(available, 2)
                    n3, n4 = candidates[j1], candidates[j2]
                    
                    new_set = current_set[:]
                    new_set[i1] = n3
                    new_set[i2] = n4
                    
                    if is_valid(new_set):
                        current_set = new_set
                        in_set[idx1] = False
                        in_set[idx2] = False
                        in_set[j1] = True
                        in_set[j2] = True
                        improved = True
                        break
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            if best_len >= 18:
                break

    return best_set
