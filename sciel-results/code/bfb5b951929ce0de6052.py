"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def get_conflicts(coloring, k):
    """
    Returns a list of conflicts.
    Conflict types:
    1. ('adj', u, v) : coloring[u] == coloring[v]
    2. ('bicolored', c1, c2, v) : A bicolored cycle exists for colors c1, c2, and v is a vertex involved.
    """
    conflicts = []
    
    # 1. Adjacency conflicts
    for u, v in edges:
        if coloring[u] == coloring[v]:
            conflicts.append(('adj', u, v))
            
    # 2. Bicolored cycle conflicts
    # Only check pairs of colors that are actually used
    used_colors = set(coloring)
    color_list = list(used_colors)
    
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            c1 = color_list[i]
            c2 = color_list[j]
            
            # Check if subgraph induced by c1, c2 has a cycle
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            has_cycle = False
            involved_vertices = []
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        has_cycle = True
                        # Collect some vertices from this component to pick a random one
                        # We can just pick u or v for now, or collect all
                        if u not in involved_vertices:
                            involved_vertices.append(u)
                        if v not in involved_vertices:
                            involved_vertices.append(v)
            
            if has_cycle:
                # Add a conflict for this pair. We'll pick a random vertex from involved_vertices later
                # But we need to know which vertices are in the cyclic part.
                # For simplicity, we can just return the pair and let the solver pick a random vertex
                # from the set of vertices with color c1 or c2 that are connected in this subgraph.
                # To keep it fast, we just flag the pair.
                conflicts.append(('bicolored', c1, c2, None))
                
    return conflicts

def validate_full(coloring):
    """Check proper coloring and no bicolored cycles."""
    k = max(coloring) + 1
    conflicts = get_conflicts(coloring, k)
    return len(conflicts) == 0

def local_search(k, deadline):
    """
    Randomized local search for a valid coloring with k colors.
    """
    # Initialize random coloring
    coloring = [random.randint(0, k - 1) for _ in range(n)]
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        conflicts = get_conflicts(coloring, k)
        
        if not conflicts:
            return coloring
            
        # Pick a random conflict
        conflict = random.choice(conflicts)
        
        if conflict[0] == 'adj':
            _, u, v = conflict
            # Recolor u to a random different color
            old_c = coloring[u]
            new_c = random.randint(0, k - 1)
            while new_c == old_c:
                new_c = random.randint(0, k - 1)
            coloring[u] = new_c
            
        elif conflict[0] == 'bicolored':
            _, c1, c2, _ = conflict
            # Find a vertex to recolor.
            # We need a vertex that is part of the bicolored cycle.
            # Let's find all vertices with color c1 or c2 that are connected in the induced subgraph.
            # To be efficient, we can just pick a random vertex with color c1 or c2 
            # and hope it helps, or better, find a specific one.
            
            # Let's find a specific vertex in the cycle.
            # Re-run union-find to find the specific edge that closed the cycle?
            # Or just pick a random vertex from the set of vertices with color c1 or c2.
            # This might be slow if it doesn't break the cycle.
            
            # Better: Pick a random edge (u,v) in the induced subgraph of c1,c2.
            # If we recolor u, it might break the cycle.
            
            candidate_vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
            if not candidate_vertices:
                continue # Should not happen
            
            v = random.choice(candidate_vertices)
            old_c = coloring[v]
            new_c = random.randint(0, k - 1)
            while new_c == old_c:
                new_c = random.randint(0, k - 1)
            coloring[v] = new_c
            
    return None

def construct():
    deadline = time.time() + 85.0
    random.seed(42) # For reproducibility in testing, but evaluator might not care
    
    # Try k = 3, 4, 5, 6
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few restarts for each k
        for restart in range(10):
            result = local_search(k, deadline)
            if result is not None:
                # Final validation
                if validate_full(result):
                    return [int(c) for c in result]
            
    # Fallback: Greedy with backtracking that checks bicolored cycles
    # This is a slower but deterministic approach for the last resort
    coloring = [-1] * n
    degree = [len(adj[v]) for v in range(n)]
    
    def backtrack(idx, k):
        if idx == n:
            return True
        v = idx
        for c in range(k):
            coloring[v] = c
            # Check adjacency
            ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    ok = False
                    break
            if ok:
                # Check bicolored cycles involving v?
                # This is expensive. For fallback, we just do proper coloring
                # and hope. But this fallback is likely to fail bicolored check.
                # So we just return a proper coloring as last resort.
                if backtrack(idx + 1, k):
                    return True
            coloring[v] = -1
        return False
    
    for k in [3, 4, 5, 6]:
        coloring = [-1] * n
        if backtrack(0, k):
            # Validate
            if validate_full(coloring):
                return [int(c) for c in coloring]
            
    # Absolute fallback: Random valid proper coloring (might have bicolored cycles, but better than invalid type)
    # Actually, if we fail all, we return a random coloring that is at least proper.
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
