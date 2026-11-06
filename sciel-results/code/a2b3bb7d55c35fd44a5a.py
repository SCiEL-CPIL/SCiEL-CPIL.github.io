import time
import random
import math
import heapq
from functools import lru_cache

def construct():
    start_time = time.time()
    limit = 85.0 # Safety margin for 90s limit
    
    n = 256
    
    # Weights: 1/i
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Build Conflict Graph
    # adj_mask[i] is a bitmask of all j such that (i,j) is an edge in G_static.
    adj_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            conflict = False
            # Check 3rd terms
            c1 = 2 * i - j
            if 1 <= c1 <= n:
                conflict = True
            c2 = 2 * j - i
            if 1 <= c2 <= n:
                conflict = True
            if (i + j) % 2 == 0:
                c3 = (i + j) // 2
                if 1 <= c3 <= n:
                    conflict = True
            
            if conflict:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Helper to check if a vertex can be added to a set represented by a bitmask
    # A vertex v can be added if it has no neighbors in the set.
    def can_add_bitmask(v, mask):
        return (mask & adj_mask[v]) == 0

    # Greedy initialization
    # Pick vertices in order of decreasing weight if they don't conflict with already picked ones.
    # This gives a good starting point for SA.
    order = list(range(1, n + 1))
    order.sort(key=lambda x: -weights[x])
    
    init_mask = 0
    init_weight = 0.0
    for v in order:
        if can_add_bitmask(v, init_mask):
            init_mask |= (1 << v)
            init_weight += weights[v]
            
    # Simulated Annealing
    best_mask = init_mask
    best_weight = init_weight
    current_mask = init_mask
    current_weight = init_weight
    
    # Parameters
    T = 0.5 # Initial temperature (scale of weight differences)
    alpha = 0.999 # Cooling rate
    min_T = 1e-5
    
    # Precompute list of vertices for random selection
    vertices = list(range(1, n + 1))
    
    while time.time() - start_time < limit:
        if T < min_T:
            break
            
        # Generate a random move: pick a vertex to flip (add or remove)
        # To improve efficiency, we can restrict moves to vertices that are "borderline"
        # i.e., neighbors of the current set or vertices that can be added.
        # However, for simplicity and robustness, let's pick any vertex.
        # Optimization: Only consider vertices that are either in the set (to remove) 
        # or can be added (to add). Removing a vertex that isn't in the set is a no-op.
        # Adding a vertex that conflicts is a no-op.
        
        # Let's pick a random vertex from 1..n
        v = random.choice(vertices)
        
        # Determine the move
        is_in_set = bool(current_mask & (1 << v))
        
        if is_in_set:
            # Try removing v
            new_mask = current_mask & ~(1 << v)
            new_weight = current_weight - weights[v]
            delta = new_weight - current_weight
        else:
            # Try adding v
            if can_add_bitmask(v, current_mask):
                new_mask = current_mask | (1 << v)
                new_weight = current_weight + weights[v]
                delta = new_weight - current_weight
            else:
                # Move not valid, skip or treat as large negative delta
                # To keep SA moving, we can just skip this iteration or pick another vertex.
                # Let's just skip to save time.
                T *= alpha
                continue
        
        # Acceptance probability
        if delta >= 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid overflow in exp
            if T > 0:
                prob = math.exp(delta / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            current_mask = new_mask
            current_weight = new_weight
            
            if current_weight > best_weight:
                best_weight = current_weight
                best_mask = current_mask
        
        # Cool down
        T *= alpha
        
    # Extract solution
    S_set = set()
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            S_set.add(i)
            
    # If SA didn't find anything (shouldn't happen), fallback
    if not S_set:
        S_set = {1}
        
    return list(S_set)

def can_add(x, S_bits):
    """
    Check if x can be added to S (represented by S_bits) without creating a 3-AP.
    x is 1-indexed.
    """
    # This function is kept for compatibility if external code calls it, 
    # but construct() uses the faster bitmask check internally.
    # Iterate through all y in S
    temp = S_bits
    while temp:
        low_bit = temp & -temp
        y = low_bit.bit_length() - 1
        
        # Check 1: y, x, z. z = 2x - y
        z = 2 * x - y
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 2: x, y, z. z = 2y - x
        z = 2 * y - x
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 3: x, z, y. z = (x+y)/2
        if (x + y) % 2 == 0:
            z = (x + y) // 2
            if 1 <= z <= 256:
                if S_bits & (1 << z):
                    return False
        
        temp &= temp - 1
        
    return True
