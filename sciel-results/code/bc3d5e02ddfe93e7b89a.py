"""Improved: entropy-guided SA with multi-start, better move selection, and refined cooling."""
import random
import math
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

def _build_edge_index():
    """Map (i,j) with i<j to a linear index 0..119."""
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx += 1
    return edge_to_idx

def _build_cycles():
    """
    For each 4-vertex subset, compute the 3 cycles.
    Each cycle is a tuple of 4 edge indices.
    """
    edge_to_idx = _build_edge_index()
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,c), max(b,c))]
        e3 = edge_to_idx[(min(c,d), max(c,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(min(a,b), max(a,b))]
        e2 = edge_to_idx[(min(b,d), max(b,d))]
        e3 = edge_to_idx[(min(d,c), max(d,c))]
        e4 = edge_to_idx[(min(c,a), max(c,a))]
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(min(a,c), max(a,c))]
        e2 = edge_to_idx[(min(c,b), max(c,b))]
        e3 = edge_to_idx[(min(b,d), max(b,d))]
        e4 = edge_to_idx[(min(d,a), max(d,a))]
        cycles.append((e1, e2, e3, e4))
    return cycles

def _build_edge_to_cycles():
    """Map each edge index to list of cycle indices it participates in."""
    cycles = _build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return edge_to_cycles

# Precompute
_EDGE_TO_IDX = _build_edge_index()
_CYCLES = _build_cycles()
_EDGE_TO_CYCLES = _build_edge_to_cycles()
NUM_CYCLES = len(_CYCLES)

def _compute_entropy(color_counts):
    """Compute Shannon entropy of color distribution (normalized by log(NUM_COLORS))."""
    ent = 0.0
    total = sum(color_counts)
    if total == 0:
        return 0.0
    for c in color_counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    max_ent = math.log(NUM_COLORS)
    return ent / max_ent if max_ent > 0 else 0.0

def _sa_run(seed, max_iter, time_budget):
    """Run one SA instance with given seed and parameters."""
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize with a valid low-diversity coloring
    colors = [0] * NUM_EDGES
    color_counts = [0] * NUM_COLORS
    color_counts[0] = NUM_EDGES
    num_distinct = 1
    violations = 0
    
    # cycle_color_counts[ci][color] = count of that color in cycle ci
    cycle_color_counts = [[0] * NUM_COLORS for _ in range(NUM_CYCLES)]
    for ci, cycle in enumerate(_CYCLES):
        for ei in cycle:
            cycle_color_counts[ci][colors[ei]] += 1
    
    # cycle_distinct[ci] = number of distinct colors in cycle ci
    cycle_distinct = [0] * NUM_CYCLES
    for ci in range(NUM_CYCLES):
        for c in range(NUM_COLORS):
            if cycle_color_counts[ci][c] > 0:
                cycle_distinct[ci] += 1
    
    violations = sum(1 for d in cycle_distinct if d == 4)
    
    best_colors = colors[:]
    best_distinct = num_distinct
    
    # Entropy-guided SA parameters
    T = 8.0
    T_min = 0.005
    cooling = 0.997  # Slower cooling for more exploration
    W_V = 5000  # Very large weight for violations
    W_D = 1     # Weight for diversity
    W_E = 0.5   # Weight for entropy bonus
    
    # Precompute list of unused colors for efficient sampling
    unused_colors = set(range(1, NUM_COLORS))
    
    for iteration in range(max_iter):
        if T < T_min or (time.time() - start_time) > time_budget:
            break
        
        # Pick a random edge
        ei = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[ei]
        
        # Choose new color with entropy-guided bias
        r = rng.random()
        if r < 0.5:
            # 50%: pick from unused colors (max entropy gain)
            if unused_colors:
                new_color = rng.choice(list(unused_colors))
            else:
                new_color = rng.randint(0, NUM_COLORS - 1)
        elif r < 0.8:
            # 30%: pick from rarely used colors (count <= 2)
            rare = [c for c in range(NUM_COLORS) if color_counts[c] <= 2]
            if rare:
                new_color = rng.choice(rare)
            else:
                new_color = rng.randint(0, NUM_COLORS - 1)
        else:
            # 20%: pick randomly from all colors
            new_color = rng.randint(0, NUM_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations using cycle_color_counts
        delta_violations = 0
        for ci in _EDGE_TO_CYCLES[ei]:
            old_distinct = cycle_distinct[ci]
            
            # After removing old_color:
            if cycle_color_counts[ci][old_color] == 1:
                temp_distinct = old_distinct - 1
            else:
                temp_distinct = old_distinct
            
            # After adding new_color:
            if cycle_color_counts[ci][new_color] == 0:
                new_distinct = temp_distinct + 1
            else:
                new_distinct = temp_distinct
            
            old_bad = (old_distinct == 4)
            new_bad = (new_distinct == 4)
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Calculate delta distinct
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1
        if color_counts[new_color] == 0:
            delta_distinct += 1
        
        # Calculate delta entropy (approximate: entropy increases when spreading)
        # When we add a new color, entropy increases; when we remove the last of a color, it decreases
        delta_entropy = 0.0
        if color_counts[new_color] == 0:
            delta_entropy += 1.0 / num_distinct  # Approximate entropy gain
        if color_counts[old_color] == 1:
            delta_entropy -= 1.0 / num_distinct  # Approximate entropy loss
        
        # Calculate delta cost (we want to minimize cost)
        delta_cost = delta_violations * W_V - delta_distinct * W_D - delta_entropy * W_E * T
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-min(delta_cost / T, 50))  # Clamp to avoid underflow
                if rng.random() < prob:
                    accept = True
                
        if accept:
            # Apply the move
            colors[ei] = new_color
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                unused_colors.add(old_color)
            color_counts[new_color] += 1
            if color_counts[new_color] == 1:
                unused_colors.discard(new_color)
            
            # Update cycle_color_counts and cycle_distinct
            for ci in _EDGE_TO_CYCLES[ei]:
                old_distinct = cycle_distinct[ci]
                
                cycle_color_counts[ci][old_color] -= 1
                if cycle_color_counts[ci][old_color] == 0:
                    temp_distinct = old_distinct - 1
                else:
                    temp_distinct = old_distinct
                
                old_new_count = cycle_color_counts[ci][new_color]
                cycle_color_counts[ci][new_color] += 1
                if old_new_count == 0:
                    new_distinct = temp_distinct + 1
                else:
                    new_distinct = temp_distinct
                
                cycle_distinct[ci] = new_distinct
            
            violations += delta_violations
            num_distinct += delta_distinct
            
            # Track best (only valid solutions)
            if violations == 0:
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
        
        T *= cooling
        
    return best_colors, best_distinct

def construct():
    """Multi-start entropy-guided simulated annealing."""
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    best_overall = None
    best_distinct_overall = 0
    
    # Multi-start with different seeds
    num_starts = 3
    per_start_time = time_limit / num_starts
    
    for s in range(num_starts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1.0:
            break
        
        seed = 42 + s * 1000
        max_iter = 150000  # Generous, will be limited by time
        
        colors, distinct = _sa_run(seed, max_iter, min(per_start_time, remaining))
        
        if distinct > best_distinct_overall:
            best_distinct_overall = distinct
            best_overall = colors
    
    if best_overall is None:
        # Fallback: all zeros (valid, 1 color)
        best_overall = [0] * NUM_EDGES
    
    return best_overall
