import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Map (u,v) to edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(len(edges))]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    def count_colors_in_k4(k4_idx, colors):
        e_list = k4s[k4_idx]
        cset = set()
        for ei in e_list:
            cset.add(colors[ei])
        return len(cset)
    
    def is_k4_valid(k4_idx, colors):
        return count_colors_in_k4(k4_idx, colors) >= 5
    
    def is_all_valid(colors):
        for k4_idx in range(len(k4s)):
            if not is_k4_valid(k4_idx, colors):
                return False
        return True
    
    # Try direct construction with (i+j) mod p for various p
    best_colors = None
    best_count = float('inf')
    
    for p in [13, 14, 15, 16, 17]:
        colors = [0] * len(edges)
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % p
        
        if is_all_valid(colors):
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    if best_colors is not None:
        # Compact labels
        unique = sorted(set(best_colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in best_colors]
    
    # If no simple formula works, use simulated annealing from (i+j)%17
    colors = [0] * len(edges)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Simulated annealing with penalty-based acceptance
    max_k = 17
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        e_idx = random.randint(0, len(edges) - 1)
        old_c = colors[e_idx]
        new_c = random.randint(0, max_k - 1)
        if new_c == old_c:
            continue
        
        # Compute penalty change: count K4s that become invalid or valid
        old_penalty = 0
        new_penalty = 0
        for k4_idx in edge_k4s[e_idx]:
            # Count colors before change
            e_list = k4s[k4_idx]
            old_colors = set(colors[ei] for ei in e_list)
            old_count = len(old_colors)
            old_penalty += (1 if old_count < 5 else 0)
            
            # Apply change temporarily
            colors[e_idx] = new_c
            new_colors = set(colors[ei] for ei in e_list)
            new_count = len(new_colors)
            new_penalty += (1 if new_count < 5 else 0)
            colors[e_idx] = old_c
        
        delta = new_penalty - old_penalty
        colors[e_idx] = new_c
        
        if delta <= 0:
            # Accept: penalty decreased or stayed same
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < 2.718281828459045 ** (-delta / T):
                nc = len(set(colors))
                if nc < best_count:
                    best_count = nc
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_c
        
        T *= cooling
    
    # Final polish: try to reduce colors used
    for _ in range(2000):
        if time.time() > deadline:
            break
        # Try to recolor an edge to reduce the total number of distinct colors
        unique_colors = set(best_colors)
        if len(unique_colors) <= 13:
            break
        e_idx = random.randint(0, len(edges) - 1)
        old_c = best_colors[e_idx]
        # Try to change to a color already in use
        other_colors = [c for c in unique_colors if c != old_c]
        if not other_colors:
            continue
        new_c = random.choice(other_colors)
        
        # Check validity
        valid = True
        for k4_idx in edge_k4s[e_idx]:
            e_list = k4s[k4_idx]
            cset = set(best_colors[ei] for ei in e_list)
            cset.discard(old_c)
            cset.add(new_c)
            if len(cset) < 5:
                valid = False
                break
        
        if valid:
            best_colors[e_idx] = new_c
            # Update unique colors
            unique_colors = set(best_colors)
    
    # Compact
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    return [mapping[c] for c in best_colors]
