import time
import random
import math

def _score_bitset(bits):
    """
    Score a 512-bit integer representing a set of 32 elements in 1..512.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    """
    count = 0
    # d can be at most 255
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count

def _set_to_bits(S):
    bits = 0
    for x in S:
        bits |= (1 << (x - 1))
    return bits

def _bits_to_set(bits):
    S = []
    for i in range(512):
        if (bits >> i) & 1:
            S.append(i + 1)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0
    best_bits = 0
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Consecutive
    candidates.append(set(range(1, 33)))
    
    # 2. Two dense blocks
    candidates.append(set(range(1, 17)) | set(range(257, 273)))
    candidates.append(set(range(1, 17)) | set(range(129, 145)))
    candidates.append(set(range(1, 17)) | set(range(193, 209)))
    
    # 3. Structured: Arithmetic progressions with different steps
    for step in [2, 3, 4, 5]:
        start = 1
        s = set()
        while len(s) < 32 and start <= 512:
            s.add(start)
            start += step
        if len(s) == 32:
            candidates.append(s)
            
    # 4. Random
    for _ in range(50):
        candidates.append(set(random.sample(range(1, 513), 32)))
        
    # 5. Hybrid: Half dense, half random
    for _ in range(20):
        s1 = set(random.sample(range(1, 100), 16))
        s2 = set(random.sample(range(100, 513), 16))
        candidates.append(s1 | s2)

    # Evaluate all candidates and pick the best one to start SA
    for c in candidates:
        if len(c) != 32:
            continue
        b = _set_to_bits(c)
        s = _score_bitset(b)
        if s > best_score:
            best_score = s
            best_bits = b
            
    current_bits = best_bits
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 5.0
    cooling = 0.9998
    min_T = 0.01
    
    steps_since_improvement = 0
    max_stagnation = 2000
    
    # Pre-compute list of elements for current_bits? 
    # It changes, so we have to update it or re-extract.
    # Re-extracting is O(32) which is fine.
    
    while time.time() < deadline:
        # Extract elements
        elements = []
        temp = current_bits
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            elements.append(idx + 1)
            temp ^= lb
            
        if not elements:
            break
            
        # Pick a random element to replace
        old_elem = random.choice(elements)
        old_bit = 1 << (old_elem - 1)
        
        # Generate a new element
        # We want a new element that is NOT in the current set
        # To avoid many collisions, we can pick from a list of empty slots?
        # Or just pick random and retry?
        # Given 32/512 occupancy, retry is efficient.
        
        new_elem = random.randint(1, 512)
        if new_elem == old_elem:
            continue
            
        # Check if new_elem is in the set
        if (current_bits >> (new_elem - 1)) & 1:
            continue
            
        new_bit = 1 << (new_elem - 1)
        
        new_bits = (current_bits & ~old_bit) | new_bit
        
        new_score = _score_bitset(new_bits)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_bits = new_bits
            current_score = new_score
            steps_since_improvement = 0
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
        else:
            steps_since_improvement += 1
            
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Restart logic
        if steps_since_improvement > max_stagnation:
            # Reheat and restart from best
            current_bits = best_bits
            current_score = best_score
            T = 5.0
            steps_since_improvement = 0
            
            # Perturb best: change 2-3 random elements
            for _ in range(3):
                # Extract elements again
                elems = []
                temp = current_bits
                while temp:
                    lb = temp & (-temp)
                    idx = lb.bit_length() - 1
                    elems.append(idx + 1)
                    temp ^= lb
                    
                if not elems:
                    break
                    
                old = random.choice(elems)
                old_b = 1 << (old - 1)
                
                new = random.randint(1, 512)
                if (current_bits >> (new - 1)) & 1:
                    continue
                    
                new_b = 1 << (new - 1)
                current_bits = (current_bits & ~old_b) | new_b
                
            current_score = _score_bitset(current_bits)
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits

    return _bits_to_set(best_bits)
