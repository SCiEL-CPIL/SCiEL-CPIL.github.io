import time
import random
import numpy as np
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    
    def is_triangle_free(adj_bits):
        """Check triangle-free using bitmasks. adj_bits[i] is a bitmask of neighbors of i."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        return False
        return True
    
    def is_3_colorable(adj_bits):
        """Check 3-colorability using backtracking with bitmasks."""
        color = [-1] * n
        color_masks = [0, 0, 0]
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            for c in range(3):
                if color_masks[c] & adj_bits[v]:
                    forbidden |= (1 << c)
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    color_masks[c] |= (1 << v)
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
                    color_masks[c] &= ~(1 << v)
            return False
        
        return backtrack(0)
    
    def get_fiedler_vector(adj_bits):
        """Compute Fiedler vector (2nd eigenvector of Laplacian) for spectral guidance."""
        # Build adjacency matrix
        A = np.zeros((n, n))
        for u in range(n):
            for v in range(n):
                if (adj_bits[u] >> v) & 1:
                    A[u][v] = 1
        
        # Compute Laplacian
        deg = np.sum(A, axis=1)
        L = np.diag(deg) - A
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            # Normalize
            norm = np.linalg.norm(fiedler)
            if norm > 0:
                fiedler = fiedler / norm
            return fiedler
        except:
            return np.zeros(n)
    
    def spectral_edge_priority(adj_bits, candidate_pairs):
        """Rank candidate edges by spectral heuristic: prioritize edges between vertices
        with large Fiedler value differences to maximize spectral gap."""
        fiedler = get_fiedler_vector(adj_bits)
        scored = []
        for u, v in candidate_pairs:
            # Score based on Fiedler value difference
            score = abs(fiedler[u] - fiedler[v])
            scored.append((score, u, v))
        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return [(u, v) for _, u, v in scored]
    
    def random_triangle_free_graph(density):
        """Generate a random triangle-free graph with given edge probability."""
        adj_bits = [0] * n
        edges = []
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if random.random() < density:
                common = adj_bits[u] & adj_bits[v]
                if not common:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append((u, v))
        
        return adj_bits, edges
    
    def add_edge_greedy(adj_bits, edges, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if (adj_bits[u] >> v) & 1:
            return False
        common = adj_bits[u] & adj_bits[v]
        if common:
            return False
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        edges.append((u, v))
        return True
    
    def remove_edge(adj_bits, edges, u, v):
        """Remove edge (u,v) from the graph."""
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        edges.remove((u, v))
    
    best_edges = []
    best_edge_count = 0
    
    random.seed(42)
    
    # Phase 1: Try various random triangle-free graphs
    for trial in range(5000):
        if time.time() > deadline - 30:
            break
        
        density = 0.2 + 0.6 * (trial % 10) / 10
        
        adj_bits, edges = random_triangle_free_graph(density)
        edge_count = len(edges)
        
        if edge_count <= best_edge_count:
            continue
        
        if not is_triangle_free(adj_bits):
            continue
        
        if not is_3_colorable(adj_bits):
            best_edges = edges[:]
            best_edge_count = edge_count
    
    # Phase 2: Spectral-guided local search - try to add edges using Fiedler vector
    if best_edges:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Get candidate edges (non-edges that don't create triangles)
            candidates = []
            for u, v in all_pairs:
                if not ((adj_bits[u] >> v) & 1):
                    common = adj_bits[u] & adj_bits[v]
                    if not common:
                        candidates.append((u, v))
            
            if not candidates:
                break
            
            # Use spectral heuristic to prioritize edges
            prioritized = spectral_edge_priority(adj_bits, candidates)
            
            for u, v in prioritized:
                if time.time() > deadline - 10:
                    break
                
                if add_edge_greedy(adj_bits, best_edges, u, v):
                    if not is_3_colorable(adj_bits):
                        improved = True
                        best_edge_count = len(best_edges)
                    else:
                        # Revert
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        best_edges.pop()
    
    # Phase 3: Try denser graphs with spectral initialization
    if time.time() < deadline - 5:
        for density in [0.8, 0.7, 0.6, 0.5]:
            if time.time() > deadline - 5:
                break
            
            # Generate multiple random graphs at this density
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                adj_bits, edges = random_triangle_free_graph(density)
                edge_count = len(edges)
                
                if edge_count <= best_edge_count:
                    continue
                
                if not is_triangle_free(adj_bits):
                    continue
                
                if not is_3_colorable(adj_bits):
                    best_edges = edges[:]
                    best_edge_count = edge_count
    
    return [[u, v] for u, v in best_edges]
