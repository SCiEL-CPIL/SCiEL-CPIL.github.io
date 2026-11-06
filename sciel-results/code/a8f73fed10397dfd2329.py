import time
import random
import numpy as np

def _score_bitset(bits):
    """
    Score a 512-bit integer representing a set of 32 elements in 1..512.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    Uses bit operations for speed.
    """
    diffs = 0
    # For each possible difference d, check if there exists a such that
    # bits has 1s at positions a-1, a+d-1, a+2d-1
    # We can use bit shifts and AND operations.
    # For a given d, the condition is:
    # (bits & (bits >> d) & (bits >> 2*d)) != 0
    # But we need to be careful about bit positions.
    # If element x is in set, bit (x-1) is set.
    # For progression a, a+d, a+2d:
    # bits has bit (a-1), bit (a+d-1), bit (a+2d-1)
    # bits >> d has bit (a-1) if bits had bit (a+d-1)
    # bits >> 2*d has bit (a-1) if bits had bit (a+2d-1)
    # So (bits & (bits >> d) & (bits >> 2*d)) will have bit (a-1) set if all three are present.
    # We need to count distinct d where this is non-zero.
    
    for d in range(1, 257):  # d can be at most 255 since a+2d <= 512 and a >= 1
        # Check if there exists a such that a, a+d, a+2d are all in set
        # a can range from 1 to 512-2*d
        # We just need to check if (bits & (bits >> d) & (bits >> 2*d)) != 0
        if (bits & (bits >> d) & (bits >> 2*d)) != 0:
            diffs += 1
    return diffs

def _set_to_bits(S):
    """Convert a set of integers in 1..512 to a 512-bit integer."""
    bits = 0
    for x in S:
        bits |= (1 << (x - 1))
    return bits

def _bits_to_set(bits):
    """Convert a 512-bit integer to a set of integers in 1..512."""
    S = set()
    for i in range(512):
        if (bits >> i) & 1:
            S.add(i + 1)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_score = 0
    best_bits = 0
    best_set = set()
    
    # Initial random sets or structured starts
    initial_sets = []
    
    # 1. Consecutive integers
    initial_sets.append(set(range(1, 33)))
    
    # 2. Random sets
    for _ in range(10):
        s = set(random.sample(range(1, 513), 32))
        initial_sets.append(s)
    
    # 3. Structured: Try some patterns
    # Dense cluster
    initial_sets.append(set(range(1, 33)))
    # Spread out
    initial_sets.append(set(range(1, 513, 16)))
    # Mix of dense and sparse
    for _ in range(5):
        s = set(random.sample(range(1, 33), 16))
        s.update(random.sample(range(33, 513), 16))
        initial_sets.append(s)
    
    for init in initial_sets:
        current_bits = _set_to_bits(init)
        current_score = _score_bitset(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_bits = current_bits
            best_set = set(init)
        
        # Local Search using bit operations
        no_improvement_count = 0
        max_no_improvement = 500
        
        while time.time() < deadline:
            if no_improvement_count > max_no_improvement:
                break
            
            # Pick a random element to remove
            # Get list of set bits
            elements = []
            temp_bits = current_bits
            while temp_bits:
                # Find lowest set bit
                lb = temp_bits & (-temp_bits)
                idx = lb.bit_length() - 1
                elements.append(idx + 1)
                temp_bits ^= lb
            
            if not elements:
                break
                
            x = random.choice(elements)
            x_bit = 1 << (x - 1)
            
            # Pick a random element to add from 1..512 not in current_set
            y = random.randint(1, 512)
            if (current_bits >> (y - 1)) & 1:
                continue
            y_bit = 1 << (y - 1)
            
            new_bits = (current_bits & ~x_bit) | y_bit
            
            new_score = _score_bitset(new_bits)
            
            if new_score > current_score:
                current_bits = new_bits
                current_score = new_score
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_bits = current_bits
                    best_set = _bits_to_set(current_bits)
            else:
                no_improvement_count += 1
            
            # Occasional restart if stuck
            if no_improvement_count > max_no_improvement * 3:
                break

    return list(best_set)
