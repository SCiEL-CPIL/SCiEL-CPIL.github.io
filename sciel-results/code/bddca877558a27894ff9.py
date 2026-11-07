import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def count_bicolored_cycles(colors):
    """Count total number of bicolored cycles (excess edges in each bicolored subgraph)."""
    color_set = set(colors)
    total_excess = 0
    proper_violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_violations += 1
    for a in color_set:
        for b in color_set:
            if a >= b:
                continue
            # Count edges between a and b, and vertices of color a or b
            edge_count = 0
            vertex_count = 0
            for u in range(N):
                if colors[u] in (a, b):
                    vertex_count += 1
            for u, v in edges:
                if (colors[u] == a and colors[v] == b) or (colors[u] == b and colors[v] == a):
                    edge_count += 1
            # A forest on vertex_count vertices has at most vertex_count - 1 edges
            # (if vertex_count > 0)
            if vertex_count > 0:
                max_forest_edges = vertex_count - 1
            else:
                max_forest_edges = 0
            excess = edge_count - max_forest_edges
            if excess > 0:
                total_excess += excess
    return proper_violations * 100 + total_excess

def is_valid(colors):
    return count_bicolored_cycles(colors) == 0

def construct():
    deadline = time.time() + 85.0
    best_score = float('inf')
    best_colors = None
    
    # Phase 1: Try structured 4-colorings
    # i % 4
    for k in range(4, 8):
        for offset in range(k):
            colors = [(i + offset) % k for i in range(N)]
            s = count_bicolored_cycles(colors)
            if s == 0:
                return [int(c) for c in colors]
            if s < best_score:
                best_score = s
                best_colors = colors[:]
        if time.time() > deadline - 5.0:
            break
    
    # Phase 2: Local search with simulated annealing
    if best_colors is None:
        best_colors = [i % 4 for i in range(N)]
        best_score = count_bicolored_cycles(best_colors)
    
    current = best_colors[:]
    current_score = best_score
    temperature = 10.0
    cooling = 0.999
    
    step = 0
    while time.time() < deadline - 2.0:
        step += 1
        # Pick a random vertex to recolor
        v = random.randint(0, N - 1)
        old_color = current[v]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Check if new_color conflicts with neighbors
        conflict = False
        for nb in adj[v]:
            if current[nb] == new_color:
                conflict = True
                break
        if conflict:
            continue
        
        current[v] = new_color
        new_score = count_bicolored_cycles(current)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < (1.0 / (1.0 + delta / temperature)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_colors = current[:]
                if best_score == 0:
                    return [int(c) for c in best_colors]
        else:
            current[v] = old_color
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    # Phase 3: More aggressive local search - try all 4 colors for each vertex
    current = best_colors[:]
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for v in range(N):
            old_color = current[v]
            for new_color in range(4):
                if new_color == old_color:
                    continue
                conflict = False
                for nb in adj[v]:
                    if current[nb] == new_color:
                        conflict = True
                        break
                if conflict:
                    continue
                current[v] = new_color
                new_score = count_bicolored_cycles(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_colors = current[:]
                    if best_score == 0:
                        return [int(c) for c in best_colors]
                    break
            current[v] = old_color
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    # Phase 4: Try 5 colors if 4 doesn't work
    for k in range(5, 8):
        if time.time() > deadline - 1.0:
            break
        colors = [i % k for i in range(N)]
        s = count_bicolored_cycles(colors)
        if s == 0:
            return [int(c) for c in colors]
        if s < best_score:
            best_score = s
            best_colors = colors[:]
    
    # Fallback: use best found or greedy
    if best_colors is not None:
        return [int(c) for c in best_colors]
    
    # Absolute fallback: proper coloring with enough colors
    colors = [0] * N
    for i in range(N):
        used = set()
        for v in adj[i]:
            if v < i:
                used.add(colors[v])
        for c in range(50):
            if c not in used:
                colors[i] = c
                break
    return [int(c) for c in colors]
