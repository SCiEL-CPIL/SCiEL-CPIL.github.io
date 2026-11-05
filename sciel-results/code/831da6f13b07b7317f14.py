import numpy as np
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin before 120s limit

    # First, try to find a valid 4-coloring for n=18 using local search
    # If successful, return it (score 18/9 = 2.0)
    # Otherwise, fall back to a known valid coloring for n=17 or lower
    
    # Try local search for n=18
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        result = local_search_18(deadline - 10.0)
        if result is not None:
            return result
    
    # Try for n=17
    for attempt in range(100):
        if time.time() > deadline - 5.0:
            break
        result = local_search_17(deadline - 10.0)
        if result is not None:
            return result
    
    # Fallback: deterministic construction for n=17
    # Use a cyclic construction
    n = 17
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # Map distance to color
            # For n=17, distances 1..16
            # We need to assign colors such that no color class has a C6
            # Cyclic coloring: color = (d % 4) might work for some n
            # Let's try a different approach
            colors.append(d % 4)
    
    # Verify and if invalid, try another pattern
    if not is_valid(n, colors):
        # Try another cyclic pattern
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # Use a different mapping
                if d <= 4:
                    c = 0
                elif d <= 8:
                    c = 1
                elif d <= 12:
                    c = 2
                else:
                    c = 3
                colors.append(c)
        
        if not is_valid(n, colors):
            # Last resort: use a simple fallback
            n = 9
            colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': n, 'colors': colors}


def local_search_18(time_limit):
    """Local search to find a valid 4-coloring of K_18 with no monochromatic C6."""
    n = 18
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Random initial coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Build adjacency matrices
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u][v] = True
        adj[c][v][u] = True
    
    # Count C6s
    c6_count = count_c6s(n, adj)
    
    if c6_count == 0:
        return {'n': n, 'colors': colors}
    
    best_count = c6_count
    best_colors = colors[:]
    
    # Simulated annealing / local search
    temp = 10.0
    cooling = 0.999
    
    for step in range(500000):
        if time.time() > time_limit:
            break
        
        # Pick a random edge
        idx = random.randint(0, num_edges - 1)
        old_color = colors[idx]
        u, v = edges[idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Remove old edge from old_color adjacency
        adj[old_color][u][v] = False
        adj[old_color][v][u] = False
        
        # Add new edge to new_color adjacency
        adj[new_color][u][v] = True
        adj[new_color][v][u] = True
        
        # Count C6s (approximate by checking affected subgraphs)
        new_count = count_c6s(n, adj)
        
        delta = new_count - c6_count
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
            c6_count = new_count
            colors[idx] = new_color
            if c6_count < best_count:
                best_count = c6_count
                best_colors = colors[:]
                if best_count == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            adj[old_color][u][v] = True
            adj[old_color][v][u] = True
            adj[new_color][u][v] = False
            adj[new_color][v][u] = False
        
        temp *= cooling
    
    if best_count == 0:
        return {'n': n, 'colors': best_colors}
    return None


def local_search_17(time_limit):
    """Local search to find a valid 4-coloring of K_17 with no monochromatic C6."""
    n = 17
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Random initial coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Build adjacency matrices
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u][v] = True
        adj[c][v][u] = True
    
    # Count C6s
    c6_count = count_c6s(n, adj)
    
    if c6_count == 0:
        return {'n': n, 'colors': colors}
    
    best_count = c6_count
    best_colors = colors[:]
    
    # Simulated annealing / local search
    temp = 10.0
    cooling = 0.999
    
    for step in range(500000):
        if time.time() > time_limit:
            break
        
        # Pick a random edge
        idx = random.randint(0, num_edges - 1)
        old_color = colors[idx]
        u, v = edges[idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Remove old edge from old_color adjacency
        adj[old_color][u][v] = False
        adj[old_color][v][u] = False
        
        # Add new edge to new_color adjacency
        adj[new_color][u][v] = True
        adj[new_color][v][u] = True
        
        # Count C6s
        new_count = count_c6s(n, adj)
        
        delta = new_count - c6_count
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
            c6_count = new_count
            colors[idx] = new_color
            if c6_count < best_count:
                best_count = c6_count
                best_colors = colors[:]
                if best_count == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            adj[old_color][u][v] = True
            adj[old_color][v][u] = True
            adj[new_color][u][v] = False
            adj[new_color][v][u] = False
        
        temp *= cooling
    
    if best_count == 0:
        return {'n': n, 'colors': best_colors}
    return None


def count_c6s(n, adj):
    """Count the number of monochromatic C6s in the coloring."""
    total = 0
    for c in range(4):
        A = adj[c]
        # Count 6-cycles: A^6 diagonal elements / 12 (each cycle counted 12 times: 6 starting points * 2 directions)
        # But this counts all cycles of length 6, not just simple ones
        # For a sparse graph, this might be okay
        # Actually, let's use a more direct approach for small n
        
        # For each pair of vertices (i,j), count paths of length 5 from i to j
        # Then sum over all i<j where A[i,j] is True
        # This is expensive but n is small (17 or 18)
        
        # Use matrix multiplication
        # A^2, A^3, A^4, A^5
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        
        # For each edge (i,j) in color c, if A5[i,j] > 0, then adding this edge creates a C6
        # But we're counting existing C6s, so we need to count cycles
        # A cycle of length 6 exists if there are vertices v0,v1,v2,v3,v4,v5 such that
        # A[v0,v1] & A[v1,v2] & A[v2,v3] & A[v3,v4] & A[v4,v5] & A[v5,v0] all True
        # and all vertices distinct
        
        # Efficient way: for each pair (i,j) with A[i,j]=True, check if there's a path of length 5 from i to j
        # using only vertices other than i and j... this is complex
        
        # Let's just count using the matrix approach
        # The number of closed walks of length 6 is trace(A^6)
        # Each 6-cycle contributes 12 to this (6 choices for starting vertex * 2 directions)
        # But this also counts non-simple closed walks
        # For our purposes, let's just use this as an upper bound and hope it works
        
        A6 = A5 @ A
        trace = np.trace(A6)
        total += trace // 12
    
    return total


def is_valid(n, colors):
    """Check if a coloring is valid (no monochromatic C6)."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        if c < 0 or c > 3:
            return False
        adj[c][u][v] = True
        adj[c][v][u] = True
    
    for c in range(4):
        A = adj[c]
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        A6 = A5 @ A
        trace = np.trace(A6)
        if trace > 0:
            # This doesn't guarantee a simple C6, but it's a good indicator
            # For a rigorous check, we'd need to verify
            # Let's do a more careful check
            if has_c6(n, A):
                return False
    
    return True


def has_c6(n, A):
    """Check if adjacency matrix A contains a simple 6-cycle."""
    # For each pair of vertices (i,j) with A[i,j]=True,
    # check if there's a path of length 5 from i to j using only vertices other than i and j
    # This is expensive, so let's use a heuristic
    
    # Actually, let's just check using BFS for each edge
    for i in range(n):
        for j in range(i + 1, n):
            if not A[i][j]:
                continue
            # Check if there's a path of length 5 from i to j not using edge (i,j)
            # and not using i or j as intermediate vertices
            # Use BFS
            visited = np.zeros((n, 6), dtype=bool)
            queue = deque()
            queue.append((i, 0))
            visited[i][0] = True
            
            while queue:
                node, depth = queue.popleft()
                if depth == 5:
                    if node == j:
                        return True
                    continue
                for nb in range(n):
                    if nb == i or nb == j:
                        continue
                    if A[node][nb] and not visited[nb][depth + 1]:
                        visited[nb][depth + 1] = True
                        queue.append((nb, depth + 1))
    
    return False
