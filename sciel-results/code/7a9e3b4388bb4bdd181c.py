import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 32
    MAX_VAL = 512
    
    # Initialize with a random set
    # Or a structured start? Random is usually fine for SA.
    # Let's start with a random permutation of 1..512, pick first 32.
    candidates = list(range(1, MAX_VAL + 1))
    random.shuffle(candidates)
    current_set = set(candidates[:N])
    current_bits = 0
    for x in current_set:
        current_bits |= (1 << x)
        
    def score_bits(bits):
        """
        Calculate the score for a bitset representation.
        Score is the number of distinct d > 0 such that
        there exists a with a, a+d, a+2d in the set.
        Using bit operations:
        (bits & (bits >> d) & (bits >> 2d)) != 0
        """
        score = 0
        # d can range from 1 to (MAX_VAL // 2) approx
        # Actually max d is such that 2d < MAX_VAL
        for d in range(1, MAX_VAL // 2 + 1):
            if (bits & (bits >> d) & (bits >> 2 * d)) != 0:
                score += 1
        return score

    current_score = score_bits(current_bits)
    best_score = current_score
    best_bits = current_bits
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    
    # We will perform swaps
    # To make it faster, we can precompute the list of elements in the set
    # and outside the set.
    
    while time.time() < deadline:
        # Generate a random move: swap x in current_set with y not in current_set
        # To avoid O(N) lookup for random element, we can keep a list
        # But maintaining a list in sync with set is tricky during SA.
        # Given N=32, iterating is cheap.
        
        # Pick random x to remove
        x = random.choice(list(current_set))
        
        # Pick random y to add
        # We can pick from 1..512 and check if in set.
        # Rejection sampling is fine.
        while True:
            y = random.randint(1, MAX_VAL)
            if y not in current_set:
                break
        
        # Apply move
        new_set = current_set.copy()
        new_set.remove(x)
        new_set.add(y)
        
        # Update bits
        # Remove x, add y
        new_bits = current_bits
        new_bits &= ~(1 << x)
        new_bits |= (1 << y)
        
        new_score = score_bits(new_bits)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = new_set
            current_bits = new_bits
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature to allow some randomization
            
    # Convert best_bits back to list of integers
    result = []
    for i in range(1, MAX_VAL + 1):
        if best_bits & (1 << i):
            result.append(i)
            
    # Ensure exactly 32 elements (should be by construction)
    if len(result) != N:
        # Fallback to seed if something went wrong
        return list(range(1, N + 1))
        
    return result
