import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge index mapping
# Edge (i, j) with i < j
# Index = sum_{k=0}^{i-1} (N-1-k) + (j-i-1)
# Sum = i*(N-1) - i*(i-1)/2
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
# A 4-cycle is defined by 4 distinct vertices {a, b, c, d}.
# There are 3 distinct cycles on these 4 vertices:
# 1. a-b-c-d-a
# 2. a-b-d-c-a
# 3. a-c-b-d-a
# We store the tuple of 4 edge indices for each cycle.
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

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check if any cycle has 4 distinct colors."""
    for c in CYCLES:
        c0 = colors[c[0]]
        c1 = colors[c[1]]
        c2 = colors[c[2]]
        c3 = colors[c[3]]
        # Check if all 4 are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initial coloring: min(i, j) which is i for i<j
    # This is known to be valid and uses N-1 = 15 colors
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
    T = 10.0  # Initial temperature
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    
    # We will pick a random edge and try to change its color
    # To a random color in 0..119
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Pick random new color
        new_color = random.randint(0, 119)
        if new_color == old_color:
            continue
            
        # Apply change
        current_colors[e] = new_color
        
        # Check validity
        if is_valid(current_colors):
            new_score = count_distinct_colors(current_colors)
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
                
            # Accept/Reject based on Simulated Annealing
            delta = new_score - current_score
            if delta >= 0:
                current_score = new_score
            else:
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_score = new_score
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
            
    return best_colors
