"""Construct a C4-free 5-coloring of K_n, maximizing n."""
import time
import random
from itertools import combinations

DEADLINE = 115.0  # safety margin before 120s kill
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _check_c4_free(n, colors):
    """Verify no color class contains a C4."""
    # Build adjacency per color
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add(j) if False else None
            idx += 1
    # Rebuild properly
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add(j) if i == j else None
            idx += 1
    # Proper adjacency
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add(j)
            idx += 1
    # For each color, check each pair of vertices has at most 1 common neighbor
    # Actually we need: for each color c, for each pair (u,v), |N_c(u) ∩ N_c(v)| <= 1
    # Wait, C4 is u-v-w-x-u, so u and w share neighbors v and x.
    # So we need: for each color c, no two vertices have 2 or more common neighbors.
    # Equivalently: for each color c, for each pair of vertices (a,b), |N_c(a) ∩ N_c(b)| <= 1
    # But actually C4 means a-b-c-d-a, so a and c share b and d as common neighbors.
    # So the condition is: for each color c, for each pair of vertices (u,v), |N_c(u) ∩ N_c(v)| <= 1.
    
    # Build proper adjacency lists
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add(j)
            idx += 1
    # Hmm, this is wrong. adj[c] should be a list of sets, one per vertex.
    adj = [set() for _ in range(5)]
    # Let me redo this properly
    adj = [set() for _ in range(5)]  # placeholder
    # Proper: adj[color][vertex] = set of neighbors
    adj = [set() for _ in range(5)]
    adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    # Check: for each color c, for each pair (u,v), |adj[c][u] & adj[c][v]| <= 1
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[c][u] & adj[c][v]) >= 2:
                    return False
    return True

def _greedy_construct(n, seed=0):
    """Greedy construction with backtracking for C4-free 5-coloring."""
    random.seed(seed)
    colors = [0] * (n * (n - 1) // 2)
    # adj[c][v] = set of neighbors of v in color c
    adj = [[set() for _ in range(n)] for _ in range(5)]
    
    def edge_idx(i, j):
        # i < j
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def can_add_color(i, j, c):
        """Check if adding edge (i,j) with color c keeps C4-free."""
        # Need |adj[c][i] & adj[c][j]| == 0
        return len(adj[c][i] & adj[c][j]) == 0
    
    def add_edge(i, j, c):
        idx = edge_idx(i, j)
        colors[idx] = c
        adj[c][i].add(j)
        adj[c][j].add(i)
    
    def remove_edge(i, j, c):
        idx = edge_idx(i, j)
        colors[idx] = -1
        adj[c][i].discard(j)
        adj[c][j].discard(i)
    
    # Process vertices in order, assign colors to edges (new_vertex, old_vertex)
    # For vertex v, we need to color edges (v, 0), (v, 1), ..., (v, v-1)
    # such that for each color c, the set of neighbors of v in color c has no two
    # elements that share a common neighbor in color c (other than v).
    
    # Actually, the constraint when adding edge (v, u) with color c is:
    # |adj[c][v] & adj[c][u]| == 0, where adj[c][v] is the set of already-colored
    # neighbors of v in color c.
    
    # Let's do a backtracking search
    def backtrack(v):
        if v >= n:
            return True
        if _elapsed() > DEADLINE:
            return False
        # For vertex v, color edges (v, 0), (v, 1), ..., (v, v-1)
        # We'll use a simple greedy with backtracking
        # For each edge (v, u) for u in 0..v-1, try colors 0..4
        # The constraint is incremental
        
        # Try to assign colors to all edges from v to previous vertices
        # Use a recursive helper
        def assign_edge(u):
            if u < 0:
                return True
            if _elapsed() > DEADLINE:
                return False
            # Try colors in random order
            color_order = list(range(5))
            random.shuffle(color_order)
            for c in color_order:
                if can_add_color(v, u, c):
                    add_edge(v, u, c)
                    if assign_edge(u - 1):
                        return True
                    remove_edge(v, u, c)
            return False
        
        return assign_edge(v - 1)
    
    if backtrack(0):
        return colors
    return None

def _local_search(n, colors, max_iter=100000):
    """Improve a coloring by swapping colors to reduce C4 violations."""
    # Count violations per color
    # For each color c, count pairs (u,v) with |N_c(u) ∩ N_c(v)| >= 2
    # This is expensive, so let's use a simpler approach:
    # Randomly swap colors of edges and accept if it doesn't increase violations
    
    # Build adjacency
    adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    
    def count_violations():
        count = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if len(adj[c][u] & adj[c][v]) >= 2:
                        count += 1
        return count
    
    def edge_idx(i, j):
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def swap_colors(i, j, c1, c2):
        """Swap colors of edge (i,j) from c1 to c2."""
        idx = edge_idx(i, j)
        # Remove from c1
        adj[c1][i].discard(j)
        adj[c1][j].discard(i)
        # Add to c2
        adj[c2][i].add(j)
        adj[c2][j].add(i)
        colors[idx] = c2
    
    # Try random edge color changes
    random.seed(42)
    for _ in range(max_iter):
        if _elapsed() > DEADLINE:
            break
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        old_c = colors[edge_idx(i, j)]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        # Try the swap
        swap_colors(i, j, old_c, new_c)
        # Check if this edge creates a C4
        bad = False
        for c in range(5):
            # Check if (i,j) in color c creates issues
            # Actually we need to check all pairs involving i or j in color c
            # This is expensive. Let's just check if the total violations decreased.
            # For simplicity, let's just check the immediate effect
            pass
        # Revert if it made things worse (simplified: just check if edge (i,j) in new_c has common neighbors)
        if len(adj[new_c][i] & adj[new_c][j]) > 0:
            # This edge itself is in a C4 configuration
            swap_colors(i, j, new_c, old_c)
        else:
            # Accept (optimistic)
            pass
    
    return colors

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with the known n=11 construction and try to extend
    # First, try greedy construction for increasing n
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try greedy construction for larger n
    for n in range(12, 28):
        if _elapsed() > DEADLINE - 10:
            break
        # Try multiple seeds
        for seed in range(50):
            if _elapsed() > DEADLINE - 10:
                break
            colors = _greedy_construct(n, seed)
            if colors is not None:
                # Verify
                if _check_c4_free(n, colors):
                    best_n = n
                    best_colors = colors
                    break  # Found valid coloring for this n, try larger
    
    # Try local search improvement on the best found
    if best_n >= 12 and _elapsed() < DEADLINE - 20:
        # Try to extend by adding vertices
        pass
    
    return {'n': best_n, 'colors': best_colors}
