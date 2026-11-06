import random
import time
import numpy as np

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def true_score(S):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        Uses bitset operations for speed.
        """
        bitset = 0
        for v in S:
            bitset |= (1 << (v - 1))
        
        count = 0
        for d in range(1, 256):
            mask = bitset & (bitset >> d) & (bitset >> (2 * d))
            if mask != 0:
                count += 1
        return count

    def fast_score(S):
        """
        Approximate score using FFT to count total 3-APs.
        Higher total 3-APs usually correlates with more distinct differences.
        """
        N = 1024
        A = np.zeros(N, dtype=np.float32)
        for v in S:
            A[v-1] = 1.0
        
        F = np.fft.fft(A)
        # Indices for F[-2k]
        indices = (-2 * np.arange(N)) % N
        F_neg2k = F[indices]
        
        # Sum F[k] * F[k] * F[-2k]
        # Note: F is complex.
        total = np.sum(F * F * F_neg2k).real / N
        return total

    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    candidates.append(list(range(1, 17)) + list(range(257, 273)))
    candidates.append(list(range(1, 17)) + list(range(129, 145)))
    
    random.seed(42)
    for _ in range(5):
        candidates.append(random.sample(range(1, 513), 32))
        
    # Pick best initial based on fast_score
    best_initial = max(candidates, key=fast_score)
    
    # Verify true score of best initial
    best_true_score = true_score(best_initial)
    best_set = best_initial[:]
    
    current_set = best_initial[:]
    current_fast_score = fast_score(current_set)
    
    start_time = time.time()
    time_limit = 110.0
    
    temperature = 5.0
    cooling = 0.9999
    min_temp = 0.01
    
    # Track top candidates by fast_score to check true_score periodically
    top_candidates = []
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        new_val = random.randint(1, 512)
        
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        new_fast_score = fast_score(current_set)
        
        delta = new_fast_score - current_fast_score
        if delta > 0:
            current_fast_score = new_fast_score
            # Add to top candidates
            top_candidates.append((new_fast_score, current_set[:]))
            if len(top_candidates) > 100:
                top_candidates.pop(0)
        else:
            if temperature > min_temp:
                prob = np.exp(delta / temperature)
                if random.random() < prob:
                    current_fast_score = new_fast_score
                else:
                    current_set[idx] = old_val
            else:
                current_set[idx] = old_val
        
        temperature *= cooling
        
        if temperature < min_temp:
            # Restart from a random top candidate
            if top_candidates:
                # Pick a random one from top 10
                idx_top = random.randint(0, min(9, len(top_candidates)-1))
                current_set = top_candidates[idx_top][1][:]
                current_fast_score = fast_score(current_set)
            else:
                current_set = best_set[:]
                current_fast_score = fast_score(current_set)
            temperature = 5.0
            # Perturb
            for _ in range(3):
                idx = random.randint(0, 31)
                current_set[idx] = random.randint(1, 512)
                while current_set[idx] in current_set[:idx] + current_set[idx+1:]:
                    current_set[idx] = random.randint(1, 512)
            current_fast_score = fast_score(current_set)
        
        # Periodically check true score of best fast_score candidate
        if int(time.time() * 10) % 5 == 0: # Every ~0.5s
            if top_candidates:
                # Check the best one in top_candidates
                best_fast, best_set_candidate = max(top_candidates, key=lambda x: x[0])
                ts = true_score(best_set_candidate)
                if ts > best_true_score:
                    best_true_score = ts
                    best_set = best_set_candidate[:]
            # Also check current
            ts = true_score(current_set)
            if ts > best_true_score:
                best_true_score = ts
                best_set = current_set[:]

    # Final check
    if top_candidates:
        best_fast, best_set_candidate = max(top_candidates, key=lambda x: x[0])
        ts = true_score(best_set_candidate)
        if ts > best_true_score:
            best_true_score = ts
            best_set = best_set_candidate[:]
    
    ts = true_score(current_set)
    if ts > best_true_score:
        best_true_score = ts
        best_set = current_set[:]

    return best_set
