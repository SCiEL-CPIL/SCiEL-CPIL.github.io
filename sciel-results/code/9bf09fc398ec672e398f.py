"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def _count_distinct_cycle_lengths(adj, n):
    """Count distinct lengths of all simple cycles in the graph."""
    lengths = set()
    
    def dfs(start, current, visited, path):
        for nb in adj[current]:
            if nb == start and len(path) >= 4:  # cycle found (min girth 4)
                lengths.add(len(path))
            elif nb not in visited and nb > start:  # avoid duplicate cycles
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, visited, path)
                path.pop()
                visited.discard(nb)
    
    for s in range(n):
        dfs(s, s, {s}, [s])
    
    return lengths

def _is_valid(edges, n=12, min_deg=3, girth=4):
    """Check if edges form a valid graph."""
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < min_deg:
            return False
    
    # Check triangle-free (girth >= 4)
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k == i:
                    continue
                if k in adj[i] and j != k:
                    # Triangle i-j-k
                    return False
    
    # Check contains at least one cycle
    # Count edges: if connected component has E >= V, it has a cycle
    visited = [False] * n
    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            vcount = 0
            ecount = 0
            while stack:
                node = stack.pop()
                vcount += 1
                ecount += len(adj[node])
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            ecount //= 2
            if ecount >= vcount:
                return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    n = 12
    min_deg = 3
    
    # Strategy: Try known good structures, then search
    # First, try two disjoint K_{3,3} components
    # K_{3,3} on vertices {0,1,2} and {3,4,5}
    edges = []
    # First K_{3,3}: parts {0,1,2} and {3,4,5}
    for u in [0,1,2]:
        for v in [3,4,5]:
            edges.append([u, v])
    # Second K_{3,3}: parts {6,7,8} and {9,10,11}
    for u in [6,7,8]:
        for v in [9,10,11]:
            edges.append([u, v])
    
    # This gives cycles of length 4 and 6, so 2 distinct lengths
    # Try to find something better
    
    # Let's try to find a graph with only 4-cycles
    # A graph where all cycles are 4-cycles must be bipartite
    # and have no "long" cycles
    
    # Try a "ladder" type structure or other constructions
    
    # Actually, let me try to search more systematically
    # For small n=12, we can try random graphs and check
    
    best_edges = edges  # fallback
    best_len_count = 2  # K_{3,3} + K_{3,3} gives {4,6}
    
    # Try to find a graph with 1 distinct cycle length
    # All cycles must be the same length
    # Most likely: all 4-cycles
    
    # A graph with only 4-cycles: each edge is in at most... 
    # Actually, let's try specific constructions
    
    # Try: two disjoint C4 components won't work (degree 2)
    # Try: a 4-regular bipartite graph where all cycles are 4?
    # That's impossible for 4-regular - there will be longer cycles
    
    # Let me try: K_{3,3} has 4 and 6 cycles
    # What about a graph that's a "thickened" C4?
    
    # Alternative: try to find any valid graph with fewer distinct lengths
    # via random search
    
    import random
    random.seed(42)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random bipartite graph with parts of size 6
        # Each vertex in part A connects to 3+ vertices in part B
        part_a = list(range(6))
        part_b = list(range(6, 12))
        
        edges_try = []
        valid = True
        for u in part_a:
            # Choose 3-5 neighbors in part_b
            num_neighbors = random.randint(3, 5)
            neighbors = random.sample(part_b, num_neighbors)
            for v in neighbors:
                edges_try.append([u, v])
        
        # Check if all vertices in part_b have degree >= 3
        deg_b = defaultdict(int)
        deg_a = defaultdict(int)
        for u, v in edges_try:
            deg_a[u] += 1
            deg_b[v] += 1
        
        if any(deg_a[u] < 3 for u in part_a) or any(deg_b[v] < 3 for v in part_b):
            continue
        
        # Check triangle-free (bipartite, so automatically triangle-free)
        # Check has a cycle
        # Count distinct cycle lengths
        adj = defaultdict(list)
        for u, v in edges_try:
            adj[u].append(v)
            adj[v].append(u)
        
        lengths = _count_distinct_cycle_lengths(adj, 12)
        
        if lengths and len(lengths) < best_len_count:
            best_len_count = len(lengths)
            best_edges = edges_try
            if best_len_count == 1:
                break  # Can't do better
    
    # Convert to proper format
    result = []
    seen = set()
    for u, v in best_edges:
        key = (min(u,v), max(u,v))
        if key not in seen:
            seen.add(key)
            result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
