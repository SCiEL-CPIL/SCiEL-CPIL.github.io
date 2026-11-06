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

def generate_structured_coloring(n, method=0, seed=None):
    """Generate a structured initial coloring using various algebraic methods."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    colors = [0] * (n * (n-1) * (n-2) // 6)
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                if method == 0:
                    # Sum mod 3
                    colors[idx] = 0 if (a + b + c) % 3 == 0 else 1
                elif method == 1:
                    # XOR-based
                    colors[idx] = (a ^ b ^ c) & 1
                elif method == 2:
                    # Product mod 2
                    colors[idx] = 1 if (a * b * c) % 2 == 0 else 0
                elif method == 3:
                    # Difference-based
                    d1 = b - a
                    d2 = c - b
                    colors[idx] = 0 if (d1 + d2) % 3 == 0 else 1
                elif method == 4:
                    # Cyclic group based
                    m = n
                    colors[idx] = 0 if ((a * b) % m + (b * c) % m + (c * a) % m) % 3 == 0 else 1
                elif method == 5:
                    # Random with bias
                    colors[idx] = rng.randint(0, 1)
                elif method == 6:
                    # Parity of sum of pairwise products
                    colors[idx] = (a*b + b*c + c*a) & 1
                else:
                    colors[idx] = rng.randint(0, 1)
                idx += 1
    return colors

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
        if n >= 34:
            max_time_for_n = 45
        elif n >= 32:
            max_time_for_n = 25
        elif n >= 30:
            max_time_for_n = 15
        else:
            max_time_for_n = 8
            
        found = False
        start_n_time = time.time()
        
        # Try multiple structured initializations
        num_methods = min(8, max(2, int((max_time_for_n - 1) / 3)))
        time_per_attempt = (max_time_for_n - 1) / num_methods
        
        for attempt in range(num_methods):
            if time.time() > deadline - 5:
                break
            if time.time() - start_n_time > max_time_for_n - 1:
                break
                
            method = attempt
            seed = attempt * 12345 + 67890
            
            time_limit = time_per_attempt
            if time.time() + time_limit > deadline - 3:
                time_limit = max(1.0, deadline - 3 - time.time())
            
            result = solve_with_initialization(n, time_limit, method, seed)
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

def solve_with_initialization(n, time_limit=5, method=0, seed=None):
    """
    Simulated Annealing with structured initial coloring.
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
    
    # Generate structured initial coloring
    colors = generate_structured_coloring(n, method, seed)
    
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
        
    # Adaptive temperature based on violation count
    T = min(2.0, max(0.5, total_violations / 50.0))
    cooling = 0.995
    best_colors = colors.copy()
    best_violations = total_violations
    
    exp = math.exp
    
    # Track recent improvements for adaptive cooling
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Update K4s
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3:
                    delta_v -= 1
            else:
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4:
                    delta_v += 1
                    
        # Update K5s
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9:
                    delta_v -= 1
            else:
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10:
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        # Acceptance with adaptive temperature
        accepted = False
        if delta_v <= 0:
            accepted = True
        else:
            if T > 0.001:
                if random.random() < exp(-delta_v / T):
                    accepted = True
        
        if accepted:
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
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
            no_improve_steps += 1
            
        # Adaptive cooling
        T *= cooling
        
        # If stuck for too long, boost temperature
        if no_improve_steps > max_no_improve:
            T = min(2.0, T * 2.0)
            no_improve_steps = 0
        
    if best_violations == 0:
        return best_colors
    return None
