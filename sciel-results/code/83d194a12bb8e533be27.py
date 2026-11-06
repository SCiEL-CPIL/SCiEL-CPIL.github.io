import time
import random
import math
from itertools import combinations

# Cache for structural data
_structure_cache = {}

def get_structure(n):
    if n in _structure_cache:
        return _structure_cache[n]
        
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        es = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(es)
        for e in es:
            edge_to_k5s[e].append(len(k5_edges) - 1)
            
    _structure_cache[n] = {
        'num_edges': num_edges,
        'edge_to_k4s': edge_to_k4s,
        'edge_to_k5s': edge_to_k5s,
        'k4_edges': k4_edges,
        'k5_edges': k5_edges,
        'num_k4s': len(k4s),
        'num_k5s': len(k5s)
    }
    return _structure_cache[n]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try to find a solution for n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n == 34:
            max_time_for_n = 45
        elif n == 33:
            max_time_for_n = 20
        elif n == 32:
            max_time_for_n = 10
        else:
            max_time_for_n = 5
            
        found = False
        start_n_time = time.time()
        
        # Try multiple restarts with different biases
        for attempt in range(20):
            if time.time() > deadline - 5 or time.time() - start_n_time >= max_time_for_n:
                break
                
            # Each attempt gets a slice of the remaining time for this n
            time_limit = 2.5
            if time.time() + time_limit > deadline - 5:
                 time_limit = max(1.0, deadline - 5 - time.time())
                 
            # Vary the bias: try different red probabilities
            # Focus on p_red around 0.6-0.7 which seems to work well
            p_red = 0.5 + 0.1 * (attempt % 7 - 3)  # Range: 0.2 to 0.8
            p_red = max(0.3, min(0.75, p_red))
            
            result = solve_fast(n, time_limit, p_red=p_red)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        # Fallback to a known small solution
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}

def solve_fast(n, time_limit=2.5, p_red=0.6):
    """
    Optimized Simulated Annealing using count-based updates.
    """
    start_time = time.time()
    
    struct = get_structure(n)
    num_edges = struct['num_edges']
    edge_to_k4s = struct['edge_to_k4s']
    edge_to_k5s = struct['edge_to_k5s']
    k4_edges = struct['k4_edges']
    k5_edges = struct['k5_edges']
    num_k4s = struct['num_k4s']
    num_k5s = struct['num_k5s']
    
    # Initial Biased Coloring
    colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    
    # Calculate initial counts
    red_counts = [0] * num_k4s
    blue_counts = [0] * num_k5s
    
    for i, e_list in enumerate(k4_edges):
        for e in e_list:
            if colors[e] == 0:
                red_counts[i] += 1
                
    for i, e_list in enumerate(k5_edges):
        for e in e_list:
            if colors[e] == 1:
                blue_counts[i] += 1
                
    total_violations = 0
    for rc in red_counts:
        if rc == 4:
            total_violations += 1
    for bc in blue_counts:
        if bc == 10:
            total_violations += 1
            
    if total_violations == 0:
        return colors.copy()
        
    # Faster cooling schedule
    T = 0.5
    cooling = 0.99
    best_colors = colors.copy()
    best_violations = total_violations
    
    # Pre-compute exp table for speed? No, delta_v is small.
    # Use math.exp for speed
    exp = math.exp
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Update K4s
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                # Red -> Blue
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3: # Was 4, now 3 (violation removed)
                    delta_v -= 1
            else:
                # Blue -> Red
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4: # Was 3, now 4 (violation added)
                    delta_v += 1
                    
        # Update K5s
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                # Blue -> Red
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9: # Was 10, now 9 (violation removed)
                    delta_v -= 1
            else:
                # Red -> Blue
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10: # Was 9, now 10 (violation added)
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        # Acceptance
        if delta_v <= 0 or random.random() < exp(-delta_v / T):
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return colors.copy()
                
        else:
            # Revert counts
            for k4_idx in edge_to_k4s[e_idx]:
                if old_color == 0:
                    red_counts[k4_idx] += 1
                else:
                    red_counts[k4_idx] -= 1
            for k5_idx in edge_to_k5s[e_idx]:
                if old_color == 1:
                    blue_counts[k5_idx] += 1
                else:
                    blue_counts[k5_idx] -= 1
                    
        T *= cooling
        
    if best_violations == 0:
        return best_colors
    return None
