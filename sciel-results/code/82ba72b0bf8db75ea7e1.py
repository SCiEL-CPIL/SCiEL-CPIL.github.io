import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin before 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_3_colorable(adj):
        """Check 3-colorability using backtracking with bitmasks and degree ordering."""
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if color c is available (no neighbor has color c)
                conflict = False
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    def edges_from_adj(adj):
        """Reconstruct edge list from adjacency bitmasks."""
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if (adj[u] >> v) & 1:
            return False
        # Check for common neighbor
        if (adj[u] & adj[v]) != 0:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    # Phase 1: Randomized maximal triangle-free graph search
    random.seed(42)
    
    while time.time() < deadline - 15:
        adj = [0] * n
        edge_count = 0
        
        # Shuffle edges for this trial
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Check if this maximal triangle-free graph is not 3-colorable
        if not is_3_colorable(adj):
            if edge_count > best_count:
                best_count = edge_count
                best_edges = edges_from_adj(adj)
    
    # Phase 2: Local search - try to add edges to improve
    if best_edges and time.time() < deadline - 5:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try adding random missing edges
        missing_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u, v in missing_edges:
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        # Check if still not 3-colorable
                        if not is_3_colorable(adj):
                            best_count += 1
                            best_edges.append([u, v])
                            improved = True
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
    
    # Fallback: If we didn't find a good graph, use the 45-edge construction
    if best_count < 45:
        edges = []
        
        # 1. Grötzsch Graph Edges
        # Cycle on V (0-4)
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        
        # Cross edges between V and W
        for i in range(5):
            w_idx_1 = 5 + (i - 1) % 5
            w_idx_2 = 5 + (i + 1) % 5
            edges.append([i, w_idx_1])
            edges.append([i, w_idx_2])
        
        # Apex Z (10) connects to all W (5-9)
        for i in range(5):
            edges.append([10, 5 + i])
        
        # 2. Extension Edges
        # New vertices X (11-15) connect to all W (5-9)
        for x in range(11, 16):
            for w in range(5, 10):
                edges.append([x, w])
        
        # Total edges: 5 (cycle) + 10 (cross) + 5 (apex) + 25 (new) = 45
        return edges
    
    return best_edges
