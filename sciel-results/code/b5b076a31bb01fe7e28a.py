"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import sys

# Increase recursion limit for DFS in set cover
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute candidate edges: only even-odd pairs not in base
    # This ensures the graph remains bipartite (triangle-free)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:  # one even, one odd
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    candidates.append((u, v))
    
    # Define the "blocks" for the set cover model
    # 12 blocks: (0,1), (2,3), ..., (22,23)
    # A block i is "covered" by an edge (u,v) if the minimum distance from any vertex in the block to u or v is <= 2
    num_blocks = n // 2
    full_mask = (1 << num_blocks) - 1
    
    # Precompute the coverage mask for each candidate edge
    edge_masks = []
    for u, v in candidates:
        mask = 0
        for b in range(num_blocks):
            # Vertices in block b are 2*b and 2*b+1
            v1 = 2 * b
            v2 = 2 * b + 1
            # Check distance from block to u or v
            # Since we are on a path with added edges, distance is shortest path in the augmented graph.
            # However, for the "connector" heuristic, we approximate distance as:
            # min(|u - v1|, |u - v2|, |v - v1|, |v - v2|) <= 2
            # This is a sufficient condition for the block to be within 2 hops IF the edge connects nearby nodes.
            # Actually, let's be more rigorous. 
            # If we add edge (u,v), the new graph has that edge.
            # Distance from block b to edge (u,v) is min(dist(b, u), dist(b, v)) in the NEW graph.
            # This is hard to precompute statically for all pairs because it depends on other added edges.
            # BUT, the strategy says: "selecting edges that create a sparse bipartite graph where every block is within distance 2 of a selected edge's endpoint".
            # This implies we can use the path distance as a lower bound or approximation.
            # If path distance > 2, it might still be covered by other edges, but for the SET COVER model to work, 
            # we need a static definition of what one edge "covers".
            # Let's define "coverage" as: The block is within path-distance 2 of u OR v.
            # If a block is covered by k edges in this sense, is it guaranteed to be within diameter 4?
            # If every block has an endpoint within distance 2, then any two blocks B1, B2 have endpoints u1, u2 with dist(B1,u1)<=2 and dist(B2,u2)<=2.
            # This doesn't directly bound dist(B1,B2).
            
            # Let's refine the model.
            # The goal is diameter <= 4.
            # If we pick a set of edges E_add, we need max_{x,y} dist(x,y) <= 4.
            # This is equivalent to: for every pair of vertices x,y, there is a path of length <= 4.
            
            # The "block coverage" idea is a heuristic. Let's stick to the prompt's suggestion:
            # "treating the path's 24 vertices as a sequence of 12 disjoint 2-vertex blocks, and selecting edges that create a sparse bipartite graph where every block is within distance 2 of a selected edge's endpoint"
            
            # Let's calculate path distance from block b to u and v.
            d_u1 = abs(u - v1)
            d_u2 = abs(u - v2)
            d_v1 = abs(v - v1)
            d_v2 = abs(v - v2)
            min_d = min(d_u1, d_u2, d_v1, d_v2)
            if min_d <= 2:
                mask |= (1 << b)
        edge_masks.append(mask)
    
    # Filter out edges that don't cover any new blocks (useless)
    # Actually, we want to solve Set Cover: min edges such that OR of masks == full_mask.
    # But does this guarantee diameter <= 4?
    # If every block is "close" to some added edge endpoint, it reduces the effective diameter.
    # Let's assume this model finds a valid solution. We will verify with BFS afterwards.
    
    # Preprocess: remove dominated edges
    # Edge A dominates Edge B if mask(A) is a superset of mask(B) and A is not B.
    # Since all edges cost 1, we can discard dominated edges.
    # However, with 132 candidates, this is O(N^2 * 2^12) which is fast.
    
    # Let's just use the raw candidates in the search, but sort by popcount descending to help pruning.
    indexed_edges = []
    for i, mask in enumerate(edge_masks):
        if mask != 0:
            indexed_edges.append((i, mask))
            
    # Sort by number of bits set (descending)
    indexed_edges.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)
    
    # Branch and Bound for Set Cover
    best_edges = None
    best_count = num_blocks + 1  # Upper bound
    
    # Recursive search
    def search(idx, current_mask, current_edges, count):
        nonlocal best_edges, best_count
        
        # Check time
        if time.time() > deadline:
            return
            
        if count >= best_count:
            return
            
        if current_mask == full_mask:
            best_count = count
            best_edges = current_edges[:]
            return
            
        if idx >= len(indexed_edges):
            return
            
        # Option 1: Skip current edge
        search(idx + 1, current_mask, current_edges, count)
        
        # Option 2: Take current edge
        i, mask = indexed_edges[idx]
        new_mask = current_mask | mask
        if new_mask != current_mask:  # Only take if it adds something
            current_edges.append(i)
            search(idx + 1, new_mask, current_edges, count + 1)
            current_edges.pop()
            
    # Start search
    search(0, 0, [], 0)
    
    if best_edges is None:
        # Fallback to a known valid solution (hub strategy) if set cover fails or times out
        # Hub 12 connected to all odd except neighbors
        fallback = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
        # Verify fallback
        if is_valid(fallback):
            return [[int(u), int(v)] for u, v in fallback]
        else:
            # Emergency: return empty? No, must be valid.
            # Try a denser fallback
            dense_fallback = []
            for u in range(0, n, 2):
                for v in range(1, n, 2):
                    if abs(u-v) > 2:
                        dense_fallback.append((u,v))
            if is_valid(dense_fallback):
                return [[int(u), int(v)] for u, v in dense_fallback]
            return []
            
    # Convert indices back to edges
    result_edges = [candidates[i] for i in best_edges]
    
    # Verify the solution
    if is_valid(result_edges):
        return [[int(u), int(v)] for u, v in result_edges]
    else:
        # If the set cover solution is invalid (diameter > 4 or triangle), 
        # we need to fix it or fall back.
        # The set cover model might produce a graph with diameter > 4.
        # Let's try to repair by adding more edges or falling back.
        
        # Try to repair by adding edges greedily to reduce diameter
        current = result_edges[:]
        # Check diameter
        while True:
            d = compute_diameter(current)
            if d <= 4:
                break
            # Add an edge that reduces diameter most
            best_gain = 0
            best_edge = None
            for u, v in candidates:
                if (u, v) in current:
                    continue
                # Estimate gain: does it connect two distant parts?
                # Simple heuristic: add edge that minimizes max distance from its endpoints
                # This is expensive. Let's just add a hub-like edge.
                # Or just fall back to the 10-edge solution if repair is too complex.
                pass
            # For now, if invalid, fall back to the 10-edge solution
            fallback = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
            if is_valid(fallback):
                return [[int(u), int(v)] for u, v in fallback]
            break
            
        return [[int(u), int(v)] for u, v in current]

def is_valid(added):
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        
    # Check triangle-free (bipartite check)
    for u, v in added:
        if (u - v) % 2 == 0:
            return False
            
    # Check no duplicates with base
    for u, v in added:
        if (min(u,v), max(u,v)) in base_edges:
            return False
            
    # Check diameter
    d = compute_diameter(added)
    return d <= 4

def compute_diameter(added):
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    for u, v in added:
        edge_set.add((min(u,v), max(u,v)))
        
    full_adj = [set() for _ in range(n)]
    for u, v in edge_set:
        full_adj[u].add(v)
        full_adj[v].add(u)
        
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in full_adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        for d in dist:
            if d == -1:
                return float('inf')
            if d > max_d:
                max_d = d
        if max_d > 4:
            return max_d
    return max_d

# EVOLVE-BLOCK-END
