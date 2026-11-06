import random
import time

def construct():
    deadline = time.time() + 110
    
    # Try to find a graph for n=40 down to n=26
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        max_restarts = 500
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # 1. Initialize a random bipartite graph (triangle-free)
            # Split vertices into two sets
            adj = [0] * n
            # Random partition
            partition = [random.random() < 0.5 for _ in range(n)]
            
            # Add random edges between partitions
            # p should be small enough to allow future edge additions within partitions
            p = 0.05 
            for u in range(n):
                if partition[u]:
                    for v in range(u + 1, n):
                        if not partition[v]:
                            if random.random() < p:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
            
            # 2. Local search: Add edges to reduce alpha
            # We want alpha < 10
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Check alpha
                # Fast check: if alpha < 10, we are done
                if not has_independent_set(adj, n, 10):
                    # Convert adj to edge list
                    edge_list = []
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            if u < v:
                                edge_list.append([u, v])
                            mask -= lsb
                    return {'n': n, 'edges': edge_list}
                
                # Find a large independent set to target
                # We just need to find ANY independent set of size 10 to break
                # But to be efficient, we can just look for safe edges
                # that are likely to help.
                
                # Strategy: Find all safe edges (no common neighbors)
                # and add one.
                # To be smarter, prioritize edges that connect vertices in a large independent set.
                
                # Find an independent set of size 10
                iso = find_independent_set(adj, n, 10)
                if not iso:
                    # Should not happen if has_independent_set returned True
                    continue
                
                # Get vertices in the independent set
                iso_vertices = [i for i in range(n) if (iso >> i) & 1]
                
                # Find safe edges
                # Safe edge (u,v): not adjacent, and (adj[u] & adj[v]) == 0
                # We prefer edges where u,v are in iso_vertices
                candidates = []
                
                # First pass: edges within iso_vertices
                for i in range(len(iso_vertices)):
                    u = iso_vertices[i]
                    for j in range(i + 1, len(iso_vertices)):
                        v = iso_vertices[j]
                        if not (adj[u] >> v) & 1:
                            if (adj[u] & adj[v]) == 0:
                                candidates.append((u, v))
                
                # If no candidates in ISO, try any safe edge
                if not candidates:
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not (adj[u] >> v) & 1:
                                if (adj[u] & adj[v]) == 0:
                                    candidates.append((u, v))
                
                if not candidates:
                    # Stuck, restart
                    break
                
                # Pick a random candidate
                u, v = random.choice(candidates)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
    # Fallback: Return the known And(9) graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def has_independent_set(adj, n, k):
    """
    Returns True if there is an independent set of size k.
    """
    return find_independent_set(adj, n, k) is not None

def find_independent_set(adj, n, k):
    """
    Returns a bitmask of an independent set of size k if exists, else None.
    Uses backtracking with bitsets for speed.
    """
    # Candidates are all vertices initially
    # We want to pick k vertices such that no two are adjacent.
    
    def backtrack(current_mask, candidates_mask):
        if bin(current_mask).count('1') == k:
            return current_mask
        if bin(candidates_mask).count('1') < k - bin(current_mask).count('1'):
            return None
        
        # Pick a vertex from candidates
        # Heuristic: pick vertex with lowest degree in induced subgraph?
        # Or just pick the first one.
        # For speed, pick the first available.
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        # New candidates are those in candidates_mask (without v) that are NOT adjacent to v
        # Non-neighbors of v in candidates_mask
        # neighbors = adj[v]
        # non_neighbors = ~neighbors
        # new_candidates = candidates_mask & ~lsb & ~adj[v]
        new_candidates = (candidates_mask & ~lsb) & ~adj[v]
        res = backtrack(current_mask | lsb, new_candidates)
        if res is not None:
            return res
        
        # Option 2: Exclude v
        res = backtrack(current_mask, candidates_mask & ~lsb)
        if res is not None:
            return res
        
        return None

    return backtrack(0, (1 << n) - 1)
