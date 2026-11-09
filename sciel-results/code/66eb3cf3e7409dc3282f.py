import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def get_edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    vertices = list(range(N))
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    cycles.append((get_edge_index(a, b), get_edge_index(b, c), get_edge_index(c, d), get_edge_index(d, a)))
                    cycles.append((get_edge_index(a, b), get_edge_index(b, d), get_edge_index(d, c), get_edge_index(c, a)))
                    cycles.append((get_edge_index(a, c), get_edge_index(c, b), get_edge_index(b, d), get_edge_index(d, a)))
    
    best_colors = None
    best_count = 0
    
    for attempt in range(10):
        colors = list(range(NUM_EDGES))
        
        for _ in range(2000):
            violated = False
            for cycle in cycles:
                c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                if len({c1, c2, c3, c4}) == 4:
                    violated = True
                    # Count frequencies of colors in the whole graph
                    # This is expensive.
                    # Just pick randomly
                    idx = random.randint(0, 3)
                    target_idx = random.randint(0, 3)
                    if idx == target_idx:
                        target_idx = (target_idx + 1) % 4
                    
                    colors[cycle[idx]] = colors[cycle[target_idx]]
                    break
            if not violated:
                break
        
        count = len(set(colors))
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    return best_colors
