import time
import random
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 16
    
    def is_triangle_free(adj_bits):
        """Check triangle-free using bitmasks. adj_bits[i] is a bitmask of neighbors of i."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    # Check if u and v share a common neighbor
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        return False
        return True
    
    def is_3_colorable(adj_bits):
        """Check 3-colorability using backtracking with bitmasks."""
        color = [-1] * n
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        # Map order index to vertex
        pos = [0] * n
        for idx, v in enumerate(order):
            pos[v] = idx
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if color c is available (no neighbor has color c)
                conflict = False
                for u in range(n):
                    if u != v and (adj_bits[v] >> u) & 1 and color[u] == c:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def random_triangle_free_graph(density):
        """Generate a random triangle-free graph with given edge probability."""
        adj_bits = [0] * n
        edges = []
        
        all_pairs = list(combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if random.random() < density:
                # Check if adding (u,v) creates a triangle
                creates_triangle = False
                for w in range(n):
                    if w != u and w != v:
                        if (adj_bits[u] >> w) & 1 and (adj_bits[v] >> w) & 1:
                            creates_triangle = True
                            break
                if not creates_triangle:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append((u, v))
        
        return adj_bits, edges
    
    def add_edge_greedy(adj_bits, edges, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if (adj_bits[u] >> v) & 1:
            return False
        for w in range(n):
            if w != u and w != v:
                if (adj_bits[u] >> w) & 1 and (adj_bits[v] >> w) & 1:
                    return False
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        edges.append((u, v))
        return True
    
    best_edges = []
    best_edge_count = 0
    
    random.seed(42)
    
    # Phase 1: Try various random triangle-free graphs
    for trial in range(2000):
        if time.time() > deadline - 20:
            break
        
        # Try different densities
        density = 0.3 + 0.4 * (trial % 5) / 5
        
        adj_bits, edges = random_triangle_free_graph(density)
        edge_count = len(edges)
        
        if edge_count <= best_edge_count:
            continue
        
        # Verify triangle-free
        if not is_triangle_free(adj_bits):
            continue
        
        # Check non-3-colorability
        if not is_3_colorable(adj_bits):
            best_edges = edges[:]
            best_edge_count = edge_count
    
    # Phase 2: Local search - try to add edges to improve
    if best_edges:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        # Try adding random edges
        all_pairs = list(combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in all_pairs:
                if not ((adj_bits[u] >> v) & 1):
                    if add_edge_greedy(adj_bits, best_edges, u, v):
                        # Check if still not 3-colorable
                        if not is_3_colorable(adj_bits):
                            improved = True
                        else:
                            # Revert
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                            best_edges.pop()
    
    return [[u, v] for u, v in best_edges]
