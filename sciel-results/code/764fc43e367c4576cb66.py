import time
import random
import numpy as np

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge indices
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles and their edge indices
# A 4-cycle is determined by 4 vertices {a,b,c,d} with a<b<c<d
# Three 4-cycles: (a,b,c,d), (a,b,d,c), (a,c,b,d)
# Edges in cycle (x,y,z,w): xy, yz, zw, wx

four_cycles = []  # list of (edge1, edge2, edge3, edge4)
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, c)
                e3 = edge_idx(c, d)
                e4 = edge_idx(d, a)
                four_cycles.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, d)
                e3 = edge_idx(d, c)
                e4 = edge_idx(c, a)
                four_cycles.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = edge_idx(a, c)
                e2 = edge_idx(c, b)
                e3 = edge_idx(b, d)
                e4 = edge_idx(d, a)
                four_cycles.append((e1, e2, e3, e4))

# For each edge, precompute which 4-cycles it participates in
edge_to_cycles = [[] for _ in range(M)]
for ci, cycle in enumerate(four_cycles):
    for e in cycle:
        edge_to_cycles[e].append(ci)

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    for cycle in four_cycles:
        if len(set(colors[e] for e in cycle)) == 4:
            return False
    return True

def check_edge_change(colors, edge, new_color, old_color):
    """Check if changing edge to new_color keeps validity."""
    for ci in edge_to_cycles[edge]:
        e1, e2, e3, e4 = four_cycles[ci]
        c1 = new_color if e1 == edge else colors[e1]
        c2 = new_color if e2 == edge else colors[e2]
        c3 = new_color if e3 == edge else colors[e3]
        c4 = new_color if e4 == edge else colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Start with all zeros (1 color)
    colors = [0] * M
    best_colors = colors[:]
    best_count = 1
    
    # Greedily add colors: for each edge, try to assign a new color
    # We'll do multiple passes
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        # Try to recolor each edge to a new color
        progress = False
        for e in range(M):
            if time.time() > deadline - 5:
                break
            old_color = colors[e]
            # Try colors from 1 upward that haven't been used much
            for new_color in range(1, 120):
                if check_edge_change(colors, e, new_color, old_color):
                    colors[e] = new_color
                    progress = True
                    break
        
        current_count = count_colors(colors)
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if not progress:
            break
    
    # Local search / hill climbing
    if time.time() < deadline - 3:
        # Try swapping colors between edges to increase diversity
        for step in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Pick a random edge and try to change its color
            e = random.randint(0, M - 1)
            old_color = colors[e]
            
            # Try a random new color
            new_color = random.randint(0, 119)
            if new_color == old_color:
                continue
            
            if check_edge_change(colors, e, new_color, old_color):
                colors[e] = new_color
                current_count = count_colors(colors)
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback: all zeros is always valid
        return [0] * M
    
    return best_colors

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
