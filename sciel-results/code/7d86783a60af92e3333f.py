"""Edge coloring of K16 to maximize distinct colors without rainbow C4."""

import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge indices
# edge_idx[i][j] for i < j gives the index in the 120-element list
edge_idx = [[-1] * N for _ in range(N)]
for idx in range(NUM_EDGES):
    # Decode idx to (i, j)
    # i ranges from 0 to N-2
    # j ranges from i+1 to N-1
    # Number of edges for vertex i is N - 1 - i
    # Cumulative: for i=0: 15 edges, i=1: 14, ...
    # Let's precompute properly
    pass

# Better: build the mapping
edge_map = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[i][j] = idx
        edge_idx[j][i] = idx
        edge_map[idx] = (i, j)
        idx += 1

# For each 4-vertex subset, precompute the three C4s (three ways to pair 4 vertices into 2 edges, 
# but a C4 uses 4 edges). Actually a C4 on vertices {a,b,c,d} can be:
# a-b-c-d-a, a-b-d-c-a, a-c-b-d-a (three distinct cycles)
# Each cycle uses 4 edges.
# We need: for each set of 4 vertices, for each of the 3 cycles, the 4 edges must not have 4 distinct colors.

# Precompute all C4s: list of tuples of 4 edge indices
c4s = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Three C4s on {a,b,c,d}:
                # Cycle 1: a-b, b-c, c-d, d-a
                c1 = (edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a])
                # Cycle 2: a-b, b-d, d-c, c-a
                c2 = (edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a])
                # Cycle 3: a-c, c-b, b-d, d-a
                c3 = (edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a])
                c4s.append(c1)
                c4s.append(c2)
                c4s.append(c3)

NUM_C4S = len(c4s)  # 1820 * 3 = 5460


def is_valid(colors):
    """Check if no C4 has 4 distinct colors."""
    for c4 in c4s:
        e1, e2, e3, e4 = c4
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        # Check if all four are distinct
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True


def count_colors(colors):
    return len(set(colors))


def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with a valid baseline: all zeros
    best = [0] * NUM_EDGES
    best_score = 1  # one color
    
    # Strategy: greedy + local search
    # 1. Greedy construction: assign colors one by one, trying to maximize distinct colors
    # 2. Local search: try to change individual edges to new colors
    
    def elapsed():
        return time.time() - start_time
    
    def try_change(colors, edge_pos, new_color):
        """Try changing edge at position to new_color. Returns True if valid."""
        old_color = colors[edge_pos]
        if old_color == new_color:
            return True
        colors[edge_pos] = new_color
        valid = is_valid(colors)
        colors[edge_pos] = old_color
        return valid
    
    def greedy_construct():
        """Build coloring greedily, trying to introduce new colors."""
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Process edges in order, try to assign a new color if possible
        for e in range(NUM_EDGES):
            # Try unused colors first (from high to low to spread out)
            found = False
            for new_color in range(MAX_COLOR, 0, -1):
                if new_color in used_colors:
                    continue
                # Try assigning new_color
                colors[e] = new_color
                if is_valid(colors):
                    used_colors.add(new_color)
                    found = True
                    break
                colors[e] = 0
            if not found:
                colors[e] = 0
        
        return colors, len(used_colors)
    
    # Run greedy
    colors, score = greedy_construct()
    if score > best_score:
        best_score = score
        best = colors[:]
    
    # Local search: try to improve by changing individual edges
    def local_search(colors, current_score, time_budget):
        """Improve coloring by trying to change edges to new colors."""
        improved = True
        colors = colors[:]
        score = current_score
        used = set(colors)
        
        while improved and elapsed() < time_budget:
            improved = False
            # Try each edge
            for e in range(NUM_EDGES):
                if elapsed() > time_budget:
                    break
                # Try to change this edge to an unused color
                old = colors[e]
                # Try unused colors
                for new_color in range(MAX_COLOR, -1, -1):
                    if new_color in used:
                        continue
                    colors[e] = new_color
                    if is_valid(colors):
                        used.add(new_color)
                        score += 1
                        improved = True
                        break
                    colors[e] = old
                # Also try to change to a different used color that might help later
                # (skip for now, keep it simple)
        
        return colors, score
    
    # Run local search
    colors, score = local_search(colors, score, deadline - 2)
    if score > best_score:
        best_score = score
        best = colors[:]
    
    # Multiple restarts with different greedy orders
    for restart in range(20):
        if elapsed() > deadline - 3:
            break
        
        # Shuffle edge order for greedy
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        for e in edges:
            found = False
            for new_color in range(MAX_COLOR, 0, -1):
                if new_color in used_colors:
                    continue
                colors[e] = new_color
                if is_valid(colors):
                    used_colors.add(new_color)
                    found = True
                    break
                colors[e] = 0
            if not found:
                colors[e] = 0
        
        score = len(used_colors)
        
        # Local search on this
        colors, score = local_search(colors, score, deadline - 2)
        
        if score > best_score:
            best_score = score
            best = colors[:]
    
    # Final verification
    if not is_valid(best):
        # Fallback to all zeros
        return [0] * NUM_EDGES
    
    return best


# Verify at module level for sanity (optional, but let's not add overhead)
# The construct() function does its own verification
