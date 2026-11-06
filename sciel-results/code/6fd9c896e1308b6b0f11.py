import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

edge_to_idx = {}
edge_list = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        edge_list.append((i, j))
        idx += 1

c4s = []
edge_to_c4s = [[] for _ in range(NUM_EDGES)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # 1. a-b-c-d-a
    e1 = edge_to_idx[(a, b)]
    e2 = edge_to_idx[(b, c)]
    e3 = edge_to_idx[(c, d)]
    e4 = edge_to_idx[(d, a)]
    c4_idx = len(c4s)
    c4s.append((e1, e2, e3, e4))
    for e in (e1, e2, e3, e4):
        edge_to_c4s[e].append(c4_idx)
        
    # 2. a-b-d-c-a
    e1 = edge_to_idx[(a, b)]
    e2 = edge_to_idx[(b, d)]
    e3 = edge_to_idx[(d, c)]
    e4 = edge_to_idx[(c, a)]
    c4_idx = len(c4s)
    c4s.append((e1, e2, e3, e4))
    for e in (e1, e2, e3, e4):
        edge_to_c4s[e].append(c4_idx)
        
    # 3. a-c-b-d-a
    e1 = edge_to_idx[(a, c)]
    e2 = edge_to_idx[(c, b)]
    e3 = edge_to_idx[(b, d)]
    e4 = edge_to_idx[(d, a)]
    c4_idx = len(c4s)
    c4s.append((e1, e2, e3, e4))
    for e in (e1, e2, e3, e4):
        edge_to_c4s[e].append(c4_idx)

def check_edge_validity(e_idx, colors):
    for c4_idx in edge_to_c4s[e_idx]:
        e1, e2, e3, e4 = c4s[c4_idx]
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def construct():
    best_colors = [0] * NUM_EDGES
    best_count = 0
    
    # Run multiple times with different random seeds/orders
    for _ in range(10):
        colors = [0] * NUM_EDGES
        used_colors = set([0])
        available_colors = list(range(1, 120))
        random.shuffle(available_colors)
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if not available_colors:
                break
            # Try to find a valid unused color
            assigned = False
            for i in range(len(available_colors)):
                c = available_colors[i]
                old_c = colors[e_idx]
                colors[e_idx] = c
                if check_edge_validity(e_idx, colors):
                    used_colors.add(c)
                    available_colors.pop(i)
                    assigned = True
                    break
                else:
                    colors[e_idx] = old_c
            
        # Local search: try to assign remaining unused colors
        improved = True
        while improved and available_colors:
            improved = False
            for e_idx in range(NUM_EDGES):
                if not available_colors:
                    break
                current_c = colors[e_idx]
                for i in range(len(available_colors)):
                    c = available_colors[i]
                    colors[e_idx] = c
                    if check_edge_validity(e_idx, colors):
                        used_colors.add(c)
                        available_colors.pop(i)
                        improved = True
                        break
                    else:
                        colors[e_idx] = current_c
                        
        count = len(used_colors)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    return best_colors
