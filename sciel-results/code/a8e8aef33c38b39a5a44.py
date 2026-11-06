"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, with index > j
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    # Strategy: K_{6,10} + 5 edges in the 10-vertex part
    # Vertices 0-5 are the "small" part, 6-15 are the "large" part
    # This gives base degrees: small=10, large=6
    # Adding edges in large part: each endpoint gets +1 degree
    # Triangles: one small vertex + two large vertices connected
    # Score = 10 + (6+a) + (6+b) where a,b are extra degrees from internal edges
    
    # Build K_{6,10}
    base_edges = [[u, v] for u in range(6) for v in range(6, 16)]
    
    # Add 5 edges in the large part (vertices 6-15)
    # To minimize max triangle degree sum, we want to spread out the extra edges
    # Each internal edge adds 1 to each endpoint's degree
    # A triangle with small vertex s and large vertices i,j has score 10 + deg(i) + deg(j)
    # We want to minimize the max over all such triangles
    
    # Try: 5 edges forming a matching (each large vertex used at most once)
    # This gives 10 vertices in large part, 5 edges = 5 pairs
    # Each endpoint gets degree 7 instead of 6
    # Max triangle sum = 10 + 7 + 7 = 24
    
    # Let's try various configurations of 5 edges among vertices 6-15
    best_edges = None
    best_score = float('inf')
    
    # Generate all possible sets of 5 edges among 10 vertices (6-15)
    # That's C(45,5) which is too many, so use local search
    
    # Start with a matching
    initial_internal = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    initial_edges = base_edges + initial_internal
    initial_score = compute_score(initial_edges)
    best_edges = initial_edges[:]
    best_score = initial_score
    
    # Local search: swap internal edges to reduce max triangle degree sum
    def local_search(edges, max_iters=50000):
        """Try to improve score via edge swaps (only among large part vertices)."""
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        # Track which edges are internal (both endpoints in 6-15)
        internal_edge_set = set()
        for u, v in current:
            if u >= 6 and v >= 6:
                internal_edge_set.add((min(u,v), max(u,v)))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Get list of internal edges
            internal_edges = [(min(u,v), max(u,v)) for u, v in current if u >= 6 and v >= 6]
            
            # Try removing one internal edge and adding another
            for i, (u1, v1) in enumerate(internal_edges):
                if time.time() > deadline - 5.0:
                    break
                
                # Try adding a random non-edge in the large part
                for attempt in range(8):
                    u2 = random.randint(6, 15)
                    v2 = random.randint(6, 15)
                    if u2 == v2:
                        continue
                    u2, v2 = min(u2, v2), max(u2, v2)
                    if (u2, v2) == (u1, v1):
                        continue
                    if (u2, v2) in internal_edge_set:
                        continue
                    # Check it's not already an edge
                    if (adj[u2] >> v2) & 1:
                        continue
                    
                    # Perform swap
                    new_internal_edge_set = internal_edge_set - {(u1, v1)} | {(u2, v2)}
                    new_adj = adj[:]
                    new_deg = deg[:]
                    
                    # Remove edge (u1, v1)
                    new_adj[u1] &= ~(1 << v1)
                    new_adj[v1] &= ~(1 << u1)
                    new_deg[u1] -= 1
                    new_deg[v1] -= 1
                    
                    # Add edge (u2, v2)
                    new_adj[u2] |= (1 << v2)
                    new_adj[v2] |= (1 << u2)
                    new_deg[u2] += 1
                    new_deg[v2] += 1
                    
                    new_score = compute_score_from_adj(new_adj, new_deg)
                    
                    if new_score < current_score:
                        # Accept
                        adj = new_adj
                        deg = new_deg
                        current_score = new_score
                        internal_edge_set = new_internal_edge_set
                        
                        # Rebuild edge list
                        current = []
                        for u, v in edges:
                            if u >= 6 and v >= 6:
                                if (min(u,v), max(u,v)) != (u1, v1):
                                    current.append([u, v])
                            else:
                                current.append([u, v])
                        current.append([u2, v2])
                        
                        improved = True
                        break
                if improved:
                    break
        
        return current, current_score
    
    # Run local search from the initial matching
    ls_edges, ls_score = local_search(best_edges, max_iters=30000)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Try multiple random perturbations
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Random set of 5 edges among vertices 6-15
        large_vertices = list(range(6, 16))
        all_possible_edges = [(a, b) for a in range(6, 16) for b in range(a+1, 16)]
        random.shuffle(all_possible_edges)
        random_internal = all_possible_edges[:5]
        
        perturbed = base_edges + [[a, b] for a, b in random_internal]
        
        if not is_valid(perturbed):
            continue
        
        pert_score = compute_score(perturbed)
        if pert_score < best_score:
            best_score = pert_score
            best_edges = perturbed[:]
        
        # Local search from perturbation
        if time.time() > deadline - 10.0:
            continue
        ls_edges2, ls_score2 = local_search(perturbed, max_iters=10000)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_edges = ls_edges2[:]
    
    # Final verification
    if not is_valid(best_edges):
        # Fallback to initial
        best_edges = base_edges + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
