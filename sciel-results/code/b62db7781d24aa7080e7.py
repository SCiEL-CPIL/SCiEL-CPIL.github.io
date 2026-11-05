"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

N2 = N * N
points = list(itertools.product(range(N), repeat=2))
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute conflict pairs
# conflicts[i] is a set of (j, k) with j < k such that i, j, k form an isosceles triangle
conflicts = [set() for _ in range(N2)]

for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        for k in range(N2):
            if i == k or j == k:
                continue
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                if j < k:
                    conflicts[i].add((j, k))
                else:
                    conflicts[i].add((k, j))

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_size = N2
    best_set = set(range(N2))
    
    while time.time() < deadline:
        order = list(range(N2))
        random.shuffle(order)
        
        current_set = set()
        current_list = []
        
        for p in order:
            can_add = True
            len_c = len(current_list)
            for i in range(len_c):
                a = current_list[i]
                for j in range(i + 1, len_c):
                    b = current_list[j]
                    if (a, b) in conflicts[p]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                current_set.add(p)
                current_list.append(p)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return [points[i] for i in best_set]
# EVOLVE-BLOCK-END
