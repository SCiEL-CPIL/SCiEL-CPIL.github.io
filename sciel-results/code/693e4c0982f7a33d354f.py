import time
from math import gcd

weights = [0.0] * 512
neighbors = [0] * 512
best_score = 0.0
best_set = []
start_time = 0.0
time_limit = 145.0  # Safety margin

def construct():
    global weights, neighbors, best_score, best_set, start_time
    
    start_time = time.time()
    
    # Initialize weights
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    # Precompute neighbors
    # neighbors[a] has bit b set if gcd(a, b) > 1
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if a == b:
                continue
            if gcd(a, b) > 1:
                mask |= (1 << b)
        neighbors[a] = mask
        
    # Initial available set: bits 1 to 511 set
    # Bit 0 is unused
    initial_available = (1 << 512) - 2
    
    best_score = 0.0
    best_set = []
    
    search(initial_available, 0.0, [])
    
    return best_set

def search(available, current_score, current_set):
    global best_score, best_set, start_time, time_limit, weights, neighbors
    
    # Check deadline
    if time.time() - start_time > time_limit:
        return
        
    # Calculate upper bound: sum of weights of all available nodes
    bound = current_score
    temp_avail = available
    while temp_avail:
        lsb = temp_avail & -temp_avail
        idx = lsb.bit_length() - 1
        bound += weights[idx]
        temp_avail ^= lsb
        
    if bound <= best_score:
        return
        
    # If no nodes available, update best
    if available == 0:
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
        return
        
    # Select node with highest weight (highest index)
    v = available.bit_length() - 1
    
    # Branch 1: Include v
    # Remove v and its neighbors from available
    new_available = available & ~(neighbors[v] | (1 << v))
    new_score = current_score + weights[v]
    new_set = current_set + [v]
    
    search(new_available, new_score, new_set)
    
    # Branch 2: Exclude v
    new_available = available & ~(1 << v)
    
    search(new_available, current_score, current_set)
