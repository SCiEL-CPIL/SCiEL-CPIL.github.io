import random
import time
import math
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Precompute edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            
    # Precompute constraints
    # constraints[e_idx] is a list of tuples (e1, e2, e3)
    # such that {e_idx, e1, e2, e3} form a 4-cycle
    constraints = [[] for _ in range(NUM_EDGES)]
    for e_idx in range(NUM_EDGES):
        u, v = edges[e_idx]
        others = [k for k in range(N) if k != u and k != v]
        for x, y in itertools.combinations(others, 2):
            # Cycle 1: u-v-x-y-u
            e1 = edge_idx(v, x)
            e2 = edge_idx(x, y)
            e3 = edge_idx(y, u)
            constraints[e_idx].append((e1, e2, e3))
            
            # Cycle 2: u-v-y-x-u
            e1 = edge_idx(v, y)
            e2 = edge_idx(y, x) # same as (x,y)
            e3 = edge_idx(x, u)
            constraints[e_idx].append((e1, e2, e3))

    # Initialize state
    # Start with a greedy attempt to get a valid state quickly
    # or just random. Random is fine for SA.
    random.seed(42)
    colors = [random.randint(0, MAX_COLORS - 1) for _ in range(NUM_EDGES)]
    
    freq = [0] * MAX_COLORS
    for c in colors:
        freq[c] += 1
        
    distinct = sum(1 for f in freq if f > 0)
    
    # Calculate initial violations
    violations = 0
    for e_idx in range(NUM_EDGES):
        c0 = colors[e_idx]
        for e1, e2, e3 in constraints[e_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            if len({c0, c1, c2, c3}) == 4:
                violations += 1
                
    # Objective: Minimize -distinct * 10000 + violations
    # We track best valid solution
    best_distinct = 0
    best_colors = colors[:] if violations == 0 else [0]*NUM_EDGES
    
    if violations == 0:
        best_distinct = distinct
        best_colors = colors[:]

    start_time = time.time()
    time_limit = 55.0
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        new_color = random.randint(0, MAX_COLORS - 1)
        
        if new_color == old_color:
            continue
            
        # Calculate delta distinct
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        # Calculate delta violations
        delta_violations = 0
        for e1, e2, e3 in constraints[e_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            # Check old state
            was_violation = (len({old_color, c1, c2, c3}) == 4)
            # Check new state
            is_violation = (len({new_color, c1, c2, c3}) == 4)
            
            if was_violation and not is_violation:
                delta_violations -= 1
            elif not was_violation and is_violation:
                delta_violations += 1
                
        # Calculate delta objective
        # Obj = -10000 * distinct + violations
        # Delta Obj = -10000 * delta_distinct + delta_violations
        delta_obj = -10000 * delta_distinct + delta_violations
        
        # Acceptance criterion
        if delta_obj < 0 or (T > 0 and random.random() < math.exp(-delta_obj / T)):
            # Accept
            colors[e_idx] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct += delta_distinct
            violations += delta_violations
            
            if violations == 0 and distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:]
                
        T *= cooling
        
    return best_colors
