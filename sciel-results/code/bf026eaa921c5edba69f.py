"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations

def edge_index(i, j, n):
    """Return index in the lexicographic edge array for edge (i,j), i<j."""
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 3) // 2 + (j - i - 1)

def build_adj(n, colors, color):
    """Build adjacency list for a given color class."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = edge_index(i, j, n)
            if colors[idx] == color:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def find_c6_fast(n, colors):
    """
    Find all monochromatic C6s. Returns list of (color, [v0,v1,v2,v3,v4,v5]).
    Uses: for each color, for each edge (u,v), check for a path of length 5 from u to v
    in the graph with (u,v) removed. We use a BFS/DFS with pruning.
    """
    c6s = []
    for color in range(4):
        adj = build_adj(n, colors, color)
        # For each edge (u,v), look for path of length 5 from u to v avoiding (u,v)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Remove edge (u,v) temporarily
                adj[u].discard(v)
                adj[v].discard(u)
                
                # BFS for path of exactly 5 edges from u to v
                # State: (node, depth, visited_frozenset)
                # Use iterative deepening DFS with memoization on (node, depth, visited)
                # For n<=20, this is feasible with pruning
                
                # DFS with backtracking
                visited = [False] * n
                visited[u] = True
                
                def dfs(node, depth):
                    if depth == 5:
                        return node == v
                    for nb in adj[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            if dfs(nb, depth + 1):
                                return True
                            visited[nb] = False
                    return False
                
                found = dfs(u, 0)
                
                # Restore edge
                adj[u].add(v)
                adj[v].add(u)
                
                if found:
                    c6s.append((color, u, v))
    return c6s

def find_c6s_detailed(n, colors):
    """Find monochromatic C6s with full cycle info for repair."""
    c6s = []
    for color in range(4):
        adj = build_adj(n, colors, color)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                adj[u].discard(v)
                adj[v].discard(u)
                
                # DFS to find a specific path
                visited = [False] * n
                path = [u]
                visited[u] = True
                
                def dfs(node, depth):
                    if depth == 5:
                        if node == v:
                            return True
                        return False
                    for nb in sorted(adj[node]):
                        if not visited[nb]:
                            visited[nb] = True
                            path.append(nb)
                            if dfs(nb, depth + 1):
                                return True
                            path.pop()
                            visited[nb] = False
                    return False
                
                found = dfs(u, 0)
                if found:
                    # path is [u, a, b, c, d, v]
                    cycle = path[:]  # u, a, b, c, d, v
                    c6s.append((color, cycle))
                
                adj[u].add(v)
                adj[v].add(u)
    return c6s

def spectral_init(n, seed=42):
    """
    Initialize coloring using Laplacian eigenvectors.
    For K_n, the Laplacian is n*I - J (J is all-ones).
    Eigenvectors: constant (eigenvalue 0) and n-1 vectors with eigenvalue n.
    We use random projections instead since K_n Laplacian is degenerate.
    
    Better approach: use random orthonormal vectors and assign edges based on
    the sign pattern of vertex projections.
    """
    rng = np.random.RandomState(seed)
    num_edges = n * (n - 1) // 2
    colors = np.zeros(num_edges, dtype=np.int32)
    
    # Generate 3 random orthonormal vectors in R^n
    # These define a 3D embedding of the vertices
    X = rng.randn(n, 3)
    # Orthonormalize
    Q, _ = np.linalg.qr(X)
    # Q is n x 3, columns are orthonormal
    
    # For each edge (i,j), assign color based on the quadrant of the
    # difference vector Q[i] - Q[j] in 3D
    # Actually, better: assign based on the direction from i to j
    # Use 8 octants -> map to 4 colors
    
    for i in range(n):
        for j in range(i + 1, n):
            d = Q[i] - Q[j]  # 3D vector
            # Determine octant: sign of each coordinate
            s0 = 0 if d[0] >= 0 else 1
            s1 = 0 if d[1] >= 0 else 1
            s2 = 0 if d[2] >= 0 else 1
            octant = s0 * 4 + s1 * 2 + s2  # 0..7
            # Map 8 octants to 4 colors
            color = octant % 4
            idx = edge_index(i, j, n)
            colors[idx] = color
    
    return colors.tolist()

def random_init(n, seed=None):
    """Random initialization."""
    rng = random.Random(seed)
    num_edges = n * (n - 1) // 2
    return [rng.randint(0, 3) for _ in range(num_edges)]

def repair_c6s(n, colors, max_iterations=5000, deadline=None):
    """
    Local search repair: find monochromatic C6s and recolor one edge to break them.
    """
    colors = colors[:]
    best_c6_count = float('inf')
    best_colors = colors[:]
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Find all C6s
        c6s = find_c6s_detailed(n, colors)
        c6_count = len(c6s)
        
        if c6_count < best_c6_count:
            best_c6_count = c6_count
            best_colors = colors[:]
        
        if c6_count == 0:
            break
        
        # Pick a random C6 and try to break it by recoloring one of its edges
        color, cycle = random.choice(c6s)
        # cycle = [v0, v1, v2, v3, v4, v5]
        # Edges in the cycle: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
        edges_in_cycle = []
        for k in range(6):
            a = cycle[k]
            b = cycle[(k + 1) % 6]
            if a > b:
                a, b = b, a
            edges_in_cycle.append((a, b))
        
        # Try recoloring each edge in the cycle to a different color
        improved = False
        for (a, b) in edges_in_cycle:
            idx = edge_index(a, b, n)
            old_color = colors[idx]
            for new_color in range(4):
                if new_color == old_color:
                    continue
                colors[idx] = new_color
                
                # Check if this reduces C6 count
                new_c6s = find_c6s_detailed(n, colors)
                new_count = len(new_c6s)
                
                if new_count < c6_count:
                    improved = True
                    break
                else:
                    colors[idx] = old_color
            
            if improved:
                break
        
        if not improved:
            # Try random recoloring of an edge in a C6
            color, cycle = random.choice(c6s)
            edges_in_cycle = []
            for k in range(6):
                a = cycle[k]
                b = cycle[(k + 1) % 6]
                if a > b:
                    a, b = b, a
                edges_in_cycle.append((a, b))
            
            a, b = random.choice(edges_in_cycle)
            idx = edge_index(a, b, n)
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            colors[idx] = new_color
            
            # Accept if not worse
            new_c6s = find_c6s_detailed(n, colors)
            if len(new_c6s) > c6_count:
                colors[idx] = old_color
            elif len(new_c6s) == c6_count and random.random() < 0.3:
                # Accept equal with some probability for diversity
                pass
            else:
                colors[idx] = old_color
    
    if best_c6_count < len(find_c6s_detailed(n, colors)):
        return best_colors
    return colors

def verify_no_c6(n, colors):
    """Verify that no color class contains a C6."""
    return len(find_c6s_detailed(n, colors)) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # First, try to find a valid coloring for n=17 using spectral init + repair
    for target_n in [17, 18, 19]:
        if time.time() > deadline - 40:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Spectral initialization
            colors = spectral_init(target_n, seed=42 + attempt * 7)
            
            # Repair
            colors = repair_c6s(target_n, colors, max_iterations=3000, deadline=deadline - 5)
            
            # Verify
            if verify_no_c6(target_n, colors):
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if found:
            # Try to go higher
            continue
        else:
            # If we couldn't fix n=17, try harder with random inits
            for attempt in range(10):
                if time.time() > deadline - 10:
                    break
                colors = random_init(target_n, seed=1000 + attempt * 13)
                colors = repair_c6s(target_n, colors, max_iterations=2000, deadline=deadline - 5)
                if verify_no_c6(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
            if not found:
                break
    
    # Fallback: ensure we have at least a valid small coloring
    if best_n < 17:
        # Try n=17 one more time with different seeds
        if time.time() < deadline - 20:
            for attempt in range(30):
                if time.time() > deadline - 5:
                    break
                colors = spectral_init(17, seed=10000 + attempt * 3)
                colors = repair_c6s(17, colors, max_iterations=2000, deadline=deadline - 5)
                if verify_no_c6(17, colors):
                    best_n = 17
                    best_colors = colors
                    break
    
    # Final verification
    if best_n >= 1:
        num_edges = best_n * (best_n - 1) // 2
        if len(best_colors) != num_edges:
            # Fix length
            best_colors = best_colors[:num_edges]
            while len(best_colors) < num_edges:
                best_colors.append(0)
        
        # Ensure colors are in {0,1,2,3}
        best_colors = [c % 4 for c in best_colors]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
