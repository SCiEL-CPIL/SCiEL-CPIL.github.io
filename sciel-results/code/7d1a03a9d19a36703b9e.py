import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                # Cycle 2: a-b, b-d, d-c, c-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                # Cycle 3: a-c, c-b, b-d, d-a
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

# Precompute for each edge, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid_fast(colors, changed_edge, old_color):
    """Check validity only for cycles containing the changed edge."""
    for ci in EDGE_CYCLES[changed_edge]:
        cyc = CYCLES[ci]
        c0 = colors[cyc[0]]
        c1 = colors[cyc[1]]
        c2 = colors[cyc[2]]
        c3 = colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    # Initial coloring: min(i, j) which is i for i<j
    current_colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            current_colors[idx] = i
            idx += 1
            
    current_score = count_distinct_colors(current_colors)
    best_colors = current_colors[:]
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 15.0  # Higher initial temperature for more exploration
    T_min = 0.01
    alpha = 0.9995  # Slower cooling for longer exploration
    
    # Track used colors for quick new-color suggestion
    used_colors = set(current_colors)
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Strategy: 70% chance to try a new color not yet used, 30% random
        if random.random() < 0.7:
            # Try to introduce a new color
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, 119)
        else:
            new_color = random.randint(0, 119)
            
        if new_color == old_color:
            continue
            
        # Apply change
        current_colors[e] = new_color
        
        # Check validity using fast incremental check
        if is_valid_fast(current_colors, e, old_color):
            new_score = count_distinct_colors(current_colors)
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
                used_colors = set(current_colors)
                
            # Accept/Reject based on Simulated Annealing
            delta = new_score - current_score
            if delta >= 0:
                current_score = new_score
                if new_color not in used_colors:
                    used_colors.add(new_color)
                if old_color not in current_colors:
                    used_colors.discard(old_color)
            else:
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_color not in used_colors:
                        used_colors.add(new_color)
                    if old_color not in current_colors:
                        used_colors.discard(old_color)
                else:
                    # Reject: revert
                    current_colors[e] = old_color
        else:
            # Invalid: revert
            current_colors[e] = old_color
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Occasionally reheat to escape local optima
            if random.random() < 0.05:
                T = 5.0
                
    return best_colors
