import random
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precomputation
edges = []
edge_index = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))
        edge_index[(i, j)] = idx
        idx += 1

# Precompute cycles for each edge
cycles_for_edge = [[] for _ in range(NUM_EDGES)]

# Generate all 4-cycles
# 4 vertices a,b,c,d
# Cycles:
# 1: a-b, b-c, c-d, d-a
# 2: a-b, b-d, d-c, c-a
# 3: a-c, c-b, b-d, d-a
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    
    # Cycle 1
    e1 = edge_index.get((a, b))
    e2 = edge_index.get((b, c))
    e3 = edge_index.get((c, d))
    e4 = edge_index.get((d, a))
    if None not in (e1, e2, e3, e4):
        cyc = (e1, e2, e3, e4)
        for e in cyc:
            cycles_for_edge[e].append(cyc)
            
    # Cycle 2
    e1 = edge_index.get((a, b))
    e2 = edge_index.get((b, d))
    e3 = edge_index.get((d, c))
    e4 = edge_index.get((c, a))
    if None not in (e1, e2, e3, e4):
        cyc = (e1, e2, e3, e4)
        for e in cyc:
            cycles_for_edge[e].append(cyc)
            
    # Cycle 3
    e1 = edge_index.get((a, c))
    e2 = edge_index.get((c, b))
    e3 = edge_index.get((b, d))
    e4 = edge_index.get((d, a))
    if None not in (e1, e2, e3, e4):
        cyc = (e1, e2, e3, e4)
        for e in cyc:
            cycles_for_edge[e].append(cyc)

def check_edge_change(colors, edge_idx, new_color):
    for cyc in cycles_for_edge[edge_idx]:
        c1 = colors[cyc[0]] if cyc[0] != edge_idx else new_color
        c2 = colors[cyc[1]] if cyc[1] != edge_idx else new_color
        c3 = colors[cyc[2]] if cyc[2] != edge_idx else new_color
        c4 = colors[cyc[3]] if cyc[3] != edge_idx else new_color
        
        # Check if 4 distinct
        if c1 == c2 or c1 == c3 or c1 == c4:
            continue
        if c2 == c3 or c2 == c4:
            continue
        if c3 == c4:
            continue
        
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_colors = [0] * NUM_EDGES
    best_distinct = 1
    best_counts = [0] * MAX_COLOR
    best_counts[0] = NUM_EDGES
    
    current_colors = [0] * NUM_EDGES
    current_counts = [0] * MAX_COLOR
    current_counts[0] = NUM_EDGES
    distinct_count = 1
    
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        e_idx = random.randint(0, NUM_EDGES - 1)
        
        # Choose new color
        if random.random() < 0.9:
            # Try to pick an unused color
            unused = [c for c in range(MAX_COLOR) if current_counts[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLOR - 1)
        else:
            new_color = random.randint(0, MAX_COLOR - 1)
            
        if new_color == current_colors[e_idx]:
            continue
            
        if check_edge_change(current_colors, e_idx, new_color):
            old_color = current_colors[e_idx]
            
            # Update counts
            current_counts[old_color] -= 1
            if current_counts[old_color] == 0:
                distinct_count -= 1
                
            current_counts[new_color] += 1
            if current_counts[new_color] == 1:
                distinct_count += 1
                
            current_colors[e_idx] = new_color
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_colors = current_colors[:]
                best_counts = current_counts[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps > max_no_improve:
                    # Restart or perturb
                    # Simple restart from best
                    current_colors = best_colors[:]
                    current_counts = best_counts[:]
                    distinct_count = best_distinct
                    no_improve_steps = 0
                    max_no_improve = int(max_no_improve * 1.5)
        
        # Small chance to do a random valid move to explore
        if random.random() < 0.05:
             # Pick random edge and random existing color
             e_idx = random.randint(0, NUM_EDGES - 1)
             new_color = random.choice([c for c in range(MAX_COLOR) if current_counts[c] > 0])
             if new_color != current_colors[e_idx]:
                 if check_edge_change(current_colors, e_idx, new_color):
                     old_color = current_colors[e_idx]
                     current_counts[old_color] -= 1
                     if current_counts[old_color] == 0: distinct_count -= 1
                     current_counts[new_color] += 1
                     if current_counts[new_color] == 1: distinct_count += 1
                     current_colors[e_idx] = new_color

    return best_colors

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
