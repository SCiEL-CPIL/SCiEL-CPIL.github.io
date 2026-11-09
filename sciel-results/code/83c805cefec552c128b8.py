"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

def _has_c4(adj, n):
    """Check if graph has a C4 using bitset intersection."""
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & adj[v]:
                # Found common neighbor, need to check for two distinct
                common = adj[u] & adj[v]
                if common & (common - 1):  # at least 2 bits set
                    return True
    return False

def _count_c4s(adj, n):
    """Count number of C4s (each counted once)."""
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common:
                # Number of pairs of common neighbors = C(k,2)
                k = bin(common).count('1')
                if k >= 2:
                    count += k * (k - 1) // 2
    return count

def _min_degree(adj, n):
    """Return minimum degree."""
    return min(bin(adj[i]).count('1') for i in range(n))

def _try_add_edge(adj, n, u, v):
    """Try adding edge (u,v), return True if no C4 created."""
    common = adj[u] & adj[v]
    if common & (common - 1):
        return False
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    return True

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_n = 39
    best_edges = []
    
    # Try constructing C4-free graphs with high minimum degree
    # Strategy: Start from complement perspective
    # We need min_degree >= n - 39, so complement max_degree <= 38
    # But we need the graph itself to be C4-free
    
    # Approach: Use structured constructions + local search
    
    # Method 1: Start with random dense graph, remove edges to break C4s
    # Method 2: Start sparse, add edges while maintaining C4-free
    
    # For n vertices, min degree >= n-39 means complement has max degree <= 38
    # Let's work in the complement: find a graph H on n vertices with max_degree(H) <= 38
    # such that G = complement(H) is C4-free
    
    # A C4 in G means 4 vertices where all 6 edges except possibly some are in G
    # Actually, C4 in G means there exist a,b,c,d with edges ab,bc,cd,da in G
    # In complement H, this means NOT(all four of ab,bc,cd,da are in H)
    # i.e., at least one of ab,bc,cd,da is NOT in H (i.e., is in G)
    
    # This is complex. Let's use direct approach:
    # Build C4-free graph with high min degree using local search
    
    for n in range(47, 38, -1):  # Try from largest down
        if time.time() > deadline:
            break
        
        target_min_deg = n - 39
        if target_min_deg < 0:
            target_min_deg = 0
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + n)
            
            # Initialize with random graph
            adj = [0] * n
            # Start with a moderately dense random graph
            p = 0.5
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Phase 1: Remove edges to eliminate all C4s
            max_iter = 2000
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                c4_count = _count_c4s(adj, n)
                if c4_count == 0:
                    break
                # Find a C4 and remove one of its edges
                # Find any edge that's in a C4
                found = False
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            common = adj[u] & adj[v]
                            if common & (common - 1):
                                # This edge is in a C4
                                _remove_edge(adj, u, v)
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
            
            # Check if C4-free
            if _has_c4(adj, n):
                continue
            
            # Phase 2: Try to add edges to increase minimum degree
            min_deg = _min_degree(adj, n)
            
            # Local search: try to add edges where helpful
            for it in range(1000):
                if time.time() > deadline:
                    break
                if min_deg >= target_min_deg:
                    break
                
                # Find vertex with minimum degree
                min_v = min(range(n), key=lambda i: bin(adj[i]).count('1'))
                current_deg = bin(adj[min_v]).count('1')
                
                # Try adding edges from min_v to non-neighbors
                improved = False
                non_neighbors = []
                for v in range(n):
                    if v != min_v and not (adj[min_v] & (1 << v)):
                        non_neighbors.append(v)
                
                random.shuffle(non_neighbors)
                for v in non_neighbors:
                    if _try_add_edge(adj, n, min_v, v):
                        improved = True
                        break
                
                if not improved:
                    # Try swapping: remove an edge from a high-degree vertex, add to min-degree
                    break
                
                min_deg = _min_degree(adj, n)
            
            # Check final result
            if _has_c4(adj, n):
                continue
            
            min_deg = _min_degree(adj, n)
            if min_deg >= target_min_deg and n > best_n:
                best_n = n
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                best_edges = edges
            
            # If we found a valid graph for this n, we can stop trying smaller n
            # But let's continue to see if we can do better
            if n >= best_n and min_deg >= target_min_deg:
                # Good, found valid for this n
                break
    
    # Also try a more structured approach: use finite geometry
    # Projective plane of order q has q^2+q+1 vertices, each degree q+1
    # For q=6: 43 vertices, degree 7 - too sparse
    # But we can take complements or other constructions
    
    # Try: complete multipartite graphs are C4-free if parts have size <= 2
    # Complete bipartite K_{a,b} is C4-free only if min(a,b) <= 1
    # So that doesn't work directly
    
    # Another approach: polarity graphs of projective planes
    # The Erdős-Rényi graph from projective plane of order q
    # Has q^2+q+1 vertices, degree q+1, and is C4-free
    # For q=6: 43 vertices, degree 7
    
    # We need min degree >= n-39. For n=43, need degree >= 4.
    # The ER graph has degree 7, so it works! But we can do better.
    
    # Let's try to augment the ER graph
    if time.time() < deadline:
        # Construct Erdős-Rényi graph from projective plane of order 6
        # Projective plane PG(2,6) has 43 points
        # Label points as (x,y,1) for x,y in GF(6) plus line at infinity
        # But GF(6) doesn't exist! 6 is not prime power.
        # Use q=5: PG(2,5) has 31 points, degree 6
        # Or q=7: PG(2,7) has 57 points - too many
        
        # Use q=5: 31 vertices, degree 6
        # Need min degree >= 31-39 = -8, so any graph works
        # But we want n as large as possible
        
        # For n=43, need degree >= 4
        # Let's just use random construction which should work
        pass

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
