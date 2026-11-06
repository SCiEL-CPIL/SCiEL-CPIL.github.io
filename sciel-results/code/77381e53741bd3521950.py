import random
import time
import bisect

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]
    
    all_primes = primes
    num_primes = len(all_primes)
    
    def get_vector(n):
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(all_primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        return tuple(vec)
        
    vectors = {x: get_vector(x) for x in range(2, 129)}
    
    def is_valid_sequence(seq):
        if not seq:
            return True
        seen = set()
        n = len(seq)
        for i in range(n):
            acc = [0] * num_primes
            for j in range(i, n):
                v = vectors[seq[j]]
                for k in range(num_primes):
                    acc[k] += v[k]
                t = tuple(acc)
                if t in seen:
                    return False
                seen.add(t)
        return True

    best_seq = list(primes)
    best_len = len(primes)
    
    # Run many trials
    # Since each trial is fast, we can run a lot.
    # But we must stop at deadline.
    
    for trial in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        current_seq = list(primes)
        remaining_composites = composites[:]
        random.shuffle(remaining_composites)
        
        for c in remaining_composites:
            # Find sorted position
            pos = bisect.bisect_left(current_seq, c)
            
            # If c is already in current_seq (shouldn't happen as we only add composites once), skip
            # But bisect_left returns index of c if it exists.
            # Since we start with primes and add composites, c is not in current_seq initially.
            # However, if we added it in a previous step of this trial, it's in the list.
            # But we iterate through `remaining_composites` which are unique.
            # So c is not in current_seq.
            
            candidate = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid_sequence(candidate):
                current_seq = candidate
                
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq
            if best_len == 127:
                break
                
    return best_seq
# EVOLVE-BLOCK-END
