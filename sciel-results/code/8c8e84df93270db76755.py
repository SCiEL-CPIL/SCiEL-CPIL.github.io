"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i*(i+1)//2 + (j - i - 1)

def construct():
    deadline = time.time() + 55
    n_edges = N*(N-1)//2
    colors = [0] * n_edges
    freq = [0] * 120
    freq[0] = n_edges
    used_set = {0}
    
    # Precompute edge_to_sets: for each edge index, list of 4-tuples (i,j,k,l) that form a 4-cycle containing this edge
    edge_to_sets = [[] for _ in range(n_edges)]
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    # Vertices are i, j, k, l
                    # Edges in the 4-cycles on these vertices are:
                    # (i,j), (j,k), (k,l), (l,i)
                    # (i,j), (j,l), (l,k), (k,i)
                    # (i,k), (k,j), (j,l), (l,i)
                    # The set of 6 unique edges among i,j,k,l is:
                    # (i,j), (i,k), (i,l), (j,k), (j,l), (k,l)
                    edges = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                    for u, v in edges:
                        idx = _edge_idx(u, v)
                        edge_to_sets[idx].append((i, j, k, l))

    best_colors = list(colors)
    best_distinct = 1

    while time.time() < deadline:
        e = random.randint(0, n_edges - 1)
        
        # Choose a color to try
        # Bias towards unused colors to increase diversity
        if len(used_set) < 120 and random.random() < 0.7:
            # Pick from unused
            # To avoid O(120) scan every time, we can just pick random and check
            # Or maintain a list of unused? 
            # Simple: pick random 0..119, if used and we wanted unused, retry or just use it?
            # Let's just pick a random color from 0..119
            c = random.randint(0, 119)
            # If it's used, and we wanted unused, we could retry, but that's inefficient.
            # Better: if we want unused, pick from [x for x in range(120) if x not in used_set]
            # This is O(120) which is fine.
            if c in used_set:
                unused = [x for x in range(120) if x not in used_set]
                if unused:
                    c = random.choice(unused)
                else:
                    c = random.choice(list(used_set))
        else:
            c = random.choice(list(used_set))

        if c == colors[e]:
            continue

        # Check if changing colors[e] to c is valid
        old_c = colors[e]
        colors[e] = c
        
        valid = True
        for i, j, k, l in edge_to_sets[e]:
            # Cycle 1: i-j-k-l
            if len({colors[_edge_idx(i,j)], colors[_edge_idx(j,k)], colors[_edge_idx(k,l)], colors[_edge_idx(l,i)]}) == 4:
                valid = False
                break
            # Cycle 2: i-j-l-k
            if len({colors[_edge_idx(i,j)], colors[_edge_idx(j,l)], colors[_edge_idx(l,k)], colors[_edge_idx(k,i)]}) == 4:
                valid = False
                break
            # Cycle 3: i-k-j-l
            if len({colors[_edge_idx(i,k)], colors[_edge_idx(k,j)], colors[_edge_idx(j,l)], colors[_edge_idx(l,i)]}) == 4:
                valid = False
                break
        
        if not valid:
            colors[e] = old_c
        else:
            # Update freq and used_set
            freq[old_c] -= 1
            if freq[old_c] == 0:
                used_set.discard(old_c)
            
            freq[c] += 1
            if freq[c] == 1:
                used_set.add(c)
            
            # Update best if distinct count increased
            if len(used_set) > best_distinct:
                best_distinct = len(used_set)
                best_colors = list(colors)

    return best_colors
# EVOLVE-BLOCK-END
