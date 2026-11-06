import random
import time
import numpy as np

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def score_set(S):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        Uses vectorized numpy operations for speed.
        """
        S_set = set(S)
        count = 0
        # For each d from 1 to 255, check if there exists a with a, a+d, a+2d in S
        for d in range(1, 256):
            # Check if any a exists such that a, a+d, a+2d all in S
            # a must be in S, a+d in S, a+2d in S
            # a ranges from 1 to 512-2d
            found = False
            for a in range(1, 513 - 2*d):
                if a in S_set and (a+d) in S_set and (a+2*d) in S_set:
                    found = True
                    break
            if found:
                count += 1
        return count
    
    def score_bitset_fast(bitset):
        """
        Fast scoring using bitset operations.
        bitset is an integer where bit i is set if (i+1) is in S.
        """
        count = 0
        for d in range(1, 256):
            if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                count += 1
        return count

    def get_bitset(S):
        bitset = 0
        for v in S:
            bitset |= (1 << (v - 1))
        return bitset

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Dense interval
    candidates.append(list(range(1, 33)))
    
    # 2. Two dense clusters
    for offset in [100, 200, 300, 400]:
        c = list(range(1, 17)) + list(range(offset + 1, offset + 17))
        candidates.append(c)
    
    # 3. Arithmetic progressions
    for start in [1, 10, 50, 100]:
        for step in [1, 2, 3, 4, 5, 7, 11, 13]:
            c = [start + i * step for i in range(32)]
            if max(c) <= 512:
                candidates.append(c)
    
    # 4. Random sets
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(1, 513), 32))
    
    # 5. Structured: mix of small and large elements
    c_struct = list(range(1, 17)) + list(range(496, 513))
    candidates.append(c_struct)
    
    # 6. Quadratic-like distribution
    c_quad = [1 + int(i * i * 0.5) for i in range(32)]
    c_quad = [x for x in c_quad if x <= 512][:32]
    while len(c_quad) < 32:
        c_quad.append(random.randint(1, 512))
    c_quad = list(set(c_quad))[:32]
    candidates.append(c_quad)

    # Score all candidates using fast bitset method
    scored_candidates = []
    for S in candidates:
        S = list(set(S))[:32]  # Ensure distinct and exactly 32
        while len(S) < 32:
            v = random.randint(1, 512)
            if v not in S:
                S.append(v)
        b = get_bitset(S)
        s = score_bitset_fast(b)
        scored_candidates.append((s, S, b))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 8 for refinement
    top_starts = scored_candidates[:8]
    
    best_score_global = top_starts[0][0]
    best_set_global = top_starts[0][1][:]
    
    start_time = time.time()
    total_time_limit = 115.0
    time_per_start = total_time_limit / len(top_starts)
    
    rng = random.Random(123)
    
    for s_score, s_set, s_bitset in top_starts:
        current_set = s_set[:]
        current_bitset = s_bitset
        current_score = s_score
        
        if time.time() - start_time > total_time_limit:
            break
            
        start_sa_time = time.time()
        time_limit_sa = time_per_start
        
        temperature = 3.0
        cooling = 0.99998
        min_temp = 0.05
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() - start_sa_time < time_limit_sa:
            # Check timeout
            if time.time() - start_time > total_time_limit:
                break
            
            # Replace one element
            idx = rng.randint(0, 31)
            old_val = current_set[idx]
            
            # Try multiple new values, pick the best
            best_new_val = old_val
            best_new_score = current_score
            best_new_bitset = current_bitset
            
            for _ in range(5):
                new_val = rng.randint(1, 512)
                if new_val == old_val or new_val in current_set:
                    continue
                
                # Update bitset
                new_bitset = current_bitset ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
                new_set = current_set[:]
                new_set[idx] = new_val
                
                new_score = score_bitset_fast(new_bitset)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                    best_new_bitset = new_bitset
            
            if best_new_val == old_val:
                # No improvement found in this batch
                no_improve_count += 1
                temperature *= cooling
                if temperature < min_temp:
                    temperature = min_temp
                continue
            
            # Apply the best move
            current_set[idx] = best_new_val
            current_bitset = best_new_bitset
            current_score = best_new_score
            
            if current_score > best_score_global:
                best_score_global = current_score
                best_set_global = current_set[:]
                no_improve_count = 0
                temperature = max(temperature * 0.95, min_temp)
            else:
                no_improve_count += 1
            
            # Cool down
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp
            
            # Reheat if stuck
            if no_improve_count > max_no_improve:
                # Perturb from best
                current_set = best_set_global[:]
                current_bitset = get_bitset(current_set)
                current_score = best_score_global
                temperature = 3.0
                no_improve_count = 0
                # Apply 3 random perturbations
                for _ in range(3):
                    idx = rng.randint(0, 31)
                    old_val = current_set[idx]
                    new_val = rng.randint(1, 512)
                    while new_val in current_set:
                        new_val = rng.randint(1, 512)
                    current_set[idx] = new_val
                    current_bitset ^= (1 << (old_val - 1)) | (1 << (new_val - 1))
                current_score = score_bitset_fast(current_bitset)

    return best_set_global
