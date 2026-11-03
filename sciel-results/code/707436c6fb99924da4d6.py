import time
import random
from collections import deque
import sys

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # Safety margin
    
    if k != 3:
        # Fallback for other k (not expected)
        return _construct_fallback(k, g, deadline)

    # Target vertex count
    # Known best for n(3,16) is 936.
    # Lower bound is 512.
    # We aim for N around 936. If we can't get girth 16 at 936 quickly, 
    # we might try slightly larger N which is easier to satisfy.
    
    N_candidates = [936, 940, 944, 948, 950, 960, 970, 980, 1000, 1024]
    
    best_n = float('inf')
    best_edges = None
    
    for N in N_candidates:
        if time.time() - t0 > deadline - 20:
            break
            
        # Try to construct a 3-regular graph of size N with girth >= g
        # Run local search for a few seconds
        n, edges = _local_search_girth(N, g, time_budget=10.0, seed=int(time.time()*1000) + N)
        
        if n is not None:
            if n < best_n:
                best_n = n
                best_edges = edges
                # If we hit the known best (936) or lower, we can stop
                if n <= 936:
                    break
    
    if best_edges is not None:
        return best_n, best_edges
        
    # If local search failed, return a random valid graph (might have short cycles, but better than error?)
    # Actually, we must return a valid graph. 
    # If we can't find high girth, we should return the best we have? 
    # The task says "no cycle shorter than g". 
    # If we fail, we must return a valid graph. 
    # So we might need to fallback to a known valid small graph or accept failure? 
    # But "invalid result... scores 0". 
    # So we must ensure validity.
    
    # Fallback: Construct a graph of size N=1024 (power of 2, easy to verify?) 
    # No, random 3-reg might fail.
    # Let's ensure the local search runs long enough or N is large enough that random girth is high.
    # Expected girth of random 3-reg graph of size N is ~ log_2(N). 
    # For N=1024, log2(1024)=10. So random girth is ~10-12. 
    # We need 16. So random search needs local optimization.
    
    # If we still have no valid graph, we return the result of a "greedy" construction that might be large but valid.
    # However, for k=3, g=16, a graph of size ~1000 is expected to be found by local search.
    
    # Emergency fallback: 
    # If we ran out of time and haven't found a valid graph, 
    # we return a random 3-reg graph and hope for the best? No, that's bad.
    # We should prioritize finding A valid graph over finding the SMALL one.
    # But the score is N. Lower is better.
    # If we don't find valid, score 0.
    
    # So, let's increase the time budget for the first candidate.
    
    # Re-run with more time if needed? 
    # The code above tries N=936 first.
    
    return best_n, best_edges

def _construct_fallback(k, g, deadline):
    # Generic construction for other k (simplified LUW or random)
    # For the purpose of this task, k=3 is the only case.
    pass

def _local_search_girth(N, g, time_budget, seed):
    """
    Find a 3-regular graph of size N with girth >= g.
    Uses random initialization and 2-opt style edge swaps to eliminate short cycles.
    """
    random.seed(seed)
    t0 = time.time()
    
    # 1. Generate Random 3-regular Graph
    # Use configuration model with rejection for multi-edges/loops? 
    # Better: Random permutation based.
    
    # Standard method for random 3-regular graph:
    # Pair stubs.
    # Ensure no loops or multi-edges.
    
    adj = [[0]*3 for _ in range(N)]
    edges = set()
    
    # Function to check if a pair is valid (no loop, no multiedge)
    def is_valid_pair(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        return (u, v) not in edges
        
    # Configuration Model
    stubs = []
    for i in range(N):
        for _ in range(3):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    # Simple pairing often creates multi-edges. 
    # We use a "swap" approach to fix.
    # 1. Pair stubs arbitrarily.
    # 2. Check for loops and multi-edges.
    # 3. Swap connections to fix them.
    
    # Alternative: Randomized Greedy with Backtracking (slow for large N)
    # Or: Use `networkx` if available? Not allowed, only stdlib/numpy/scipy.
    
    # Let's use a fast heuristic for random 3-reg generation:
    # 1. Assign random stubs.
    # 2. Pair them.
    # 3. Identify invalid pairs (u,u) or multiple (u,v).
    # 4. Perform edge swaps to resolve.
    
    # This can be complex. 
    # Simpler: Start with a valid 3-regular graph and randomize.
    # Or just use a "random walk" on the graph space.
    
    # Let's try: 
    # 1. Build a cycle 0-1-...-N-0. (2-regular)
    # 2. Add a perfect matching on odd/even vertices to make it 3-regular.
    # This is always simple and 3-regular.
    
    for i in range(N):
        adj[i][0] = (i - 1) % N
        adj[i][1] = (i + 1) % N
        
    # Now add the 3rd edge (matching)
    # Randomly pair up vertices not adjacent in the cycle?
    # Just a random permutation pairing.
    unpaired = list(range(N))
    random.shuffle(unpaired)
    
    # This creates a specific structure. 
    # To get a "random" feel, we should shuffle the matching more thoroughly.
    
    # Let's just use the cycle + random matching.
    # Then start optimizing.
    
    # Build edge list
    edge_list = []
    adj = [[(i-1)%N, (i+1)%N] for _ in range(N)]
    
    # Pair up vertices for the 3rd edge
    # To avoid multi-edges with the cycle, we must be careful.
    # But since we are going to optimize (swap edges), we just need a starting point.
    # A random 3-reg graph is better.
    
    # Let's use a simpler random 3-reg generator:
    # 1. Stub list
    # 2. Shuffle
    # 3. Pair
    # 4. Fix loops/multi-edges by swapping.
    
    def generate_random_3reg(N):
        # Returns adj list
        stubs = [i for i in range(N) for _ in range(3)]
        random.shuffle(stubs)
        
        # Pair stubs
        pairs = []
        for i in range(0, 2*N*3, 2):
            pairs.append((stubs[i], stubs[i+1]))
            
        # Build initial edges
        edges = set()
        adj = [[] for _ in range(N)]
        
        # We will process pairs and fix collisions
        # This is tricky. Let's use a different method.
        
        # Method: Randomly assign 3 distinct neighbors to each vertex, 
        # then fix the "2-regular" constraint? No, degrees must be exactly 3.
        
        # Okay, let's use the "Swap on Invalid" strategy:
        # 1. Create stubs.
        # 2. Shuffle.
        # 3. Connect stubs[i] to stubs[i+1].
        # 4. Check for loops and multi-edges.
        # 5. If found, swap endpoints of two edges to fix.
        
        edge_endpoints = []
        for i in range(N*3):
            u, v = stubs[i], stubs[i+1] if i+1 < len(stubs) else stubs[0]
            # This pairing logic is flawed. 
            pass
        
        # Let's stick to the Cycle + Matching base case. It is valid 3-regular.
        # Then we apply random edge swaps to "mix" the graph before optimization.
        
        # Base Graph: Cycle + Random Perfect Matching
        # The matching must not connect adjacent vertices in the cycle.
        
        adj = [[(i-1)%N, (i+1)%N] for _ in range(N)]
        remaining = list(range(N))
        random.shuffle(remaining)
        
        # This matching might create multi-edges if remaining[i] is neighbor of remaining[j] in cycle.
        # We need to pick a matching from the complement of the cycle.
        
        # Simple approach:
        # Pick i, then pick j such that j is not i, i-1, i+1.
        
        used = [False]*N
        match_edges = []
        for i in range(N):
            if used[i]: continue
            candidates = [j for j in range(N) if not used[j] and j != i and abs(j-i) != 1 and abs((j-1)%N - i) != 1 and abs(j - (i+1)%N) != 1]
            # Note: Cycle neighbors are i-1 and i+1 mod N.
            # So j cannot be (i-1)%N or (i+1)%N.
            candidates = [j for j in range(N) if not used[j] and j != (i-1)%N and j != (i+1)%N]
            
            if not candidates:
                # Fallback: restart matching generation
                return generate_random_3reg(N)
                
            j = random.choice(candidates)
            adj[i].append(j)
            adj[j].append(i)
            used[i] = True
            used[j] = True
            match_edges.append((i, j))
            
        # Check validity
        # Degrees should be 3.
        # No loops/multi-edges?
        # The candidates filter prevented loops and cycle-multi-edges.
        # But it might prevent a valid matching if the complement graph has no perfect matching?
        # For N=936, the complement of a cycle is dense. Perfect matching exists.
        
        return adj, [] # Return adj only
        
    adj, _ = generate_random_3reg(N)
    
    # 2. Optimization Loop
    # Find short cycles, try to eliminate them.
    
    def find_shortest_cycle(adj, max_len):
        """Returns the cycle as a list of edges if found, else None."""
        # BFS with parent tracking for non-backtracking?
        # Simpler: For each edge (u,v), check if there is a path u-v-...-u of length < max_len.
        # This is O(N * (max_len^N))? No.
        
        # Efficient check for girth < g:
        # For each vertex, do BFS up to depth (g-1)/2.
        # If we find a node already visited at a smaller depth, we have a cycle.
        
        min_girth = g
        max_dist = (min_girth - 1) // 2
        
        for start in range(N):
            dist = [-1]*N
            parent = [-1]*N
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a connection
                        # Cycle length = du + dist[v] + 1
                        # But we must ensure it's not the edge we just came from? 
                        # No, if v is visited, it means there is another path.
                        # If the path is distinct, it forms a cycle.
                        cl = du + dist[v] + 1
                        if cl < min_girth:
                            # Reconstruct cycle
                            path1 = []
                            cur = u
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            
                            path2 = []
                            cur = v
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            
                            # Find LCA? 
                            # Just return existence for now, or the cycle itself.
                            # We need the cycle to perform swaps.
                            # Let's return the two paths.
                            return path1, path2, cl
                            
        return None

    def get_cycle_from_paths(path1, path2, N):
        # Find common ancestor (LCA) in the tree rooted at start? 
        # The paths go from LCA to u and LCA to v.
        # Set of vertices in path1
        set1 = set(path1)
        lca = None
        for node in path2:
            if node in set1:
                lca = node
                break
        if lca is None:
            return None
            
        # Cycle is lca -> ... -> u -> v -> ... -> lca
        # But we need to ensure we don't traverse back in the tree incorrectly.
        # Path1: lca...u
        # Path2: lca...v
        # Edge (u,v) closes it.
        
        idx_u = path1.index(u) if 'u' in dir() else len(path1)-1 # u is last of path1
        # Actually path1 ends at u.
        # path2 ends at v.
        
        # Construct cycle nodes
        c1 = path1[:len(path1)] # lca to u
        # Wait, path1 is from LCA to u? 
        # In BFS, parent points to start.
        # So path is Start -> ... -> LCA -> ... -> u.
        # We need the segment from LCA to u.
        
        # Let's re-extract.
        # The function `find_shortest_cycle` returned `path1` (list of nodes from start to u) and `path2` (start to v).
        # We found `lca`.
        
        # Segment 1: lca ... u
        idx_lca_in_p1 = path1.index(lca)
        seg1 = path1[idx_lca_in_p1:]
        
        idx_lca_in_p2 = path2.index(lca)
        seg2 = path2[idx_lca_in_p2:]
        
        # The cycle is seg1 + list(reversed(seg2[1:])) + seg1[0] ? 
        # Nodes: lca, ..., u, v, ..., lca
        # Edges are consecutive.
        cycle_nodes = seg1 + list(reversed(seg2[1:]))
        # Return edges
        edges = []
        for i in range(len(cycle_nodes)):
            u, v = cycle_nodes[i], cycle_nodes[(i+1)%len(cycle_nodes)]
            if (u,v) in adj[u] or (v,u) in adj[u]:
                edges.append(tuple(sorted((u,v))))
            else:
                # This path might not be simple or valid in terms of edges? 
                # BFS ensures paths are valid edges.
                pass
        return edges

    # Main Loop
    iterations = 0
    max_iterations = 2000
    
    while time.time() - t0 < time_budget and iterations < max_iterations:
        iterations += 1
        
        cycle_info = find_shortest_cycle(adj, g)
        if not cycle_info:
            # Valid graph found
            # Build edge list
            edge_list = set()
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edge_list.add((u, v))
            return N, list(edge_list)
            
        path1, path2, cl = cycle_info
        # We need the actual edges in the cycle to attempt swaps.
        # This reconstruction is complex and error-prone.
        
        # Simplified Swap Strategy:
        # Pick two edges in the graph, swap their endpoints.
        # Check if girth improved.
        
        # To speed up, we can just do random edge swaps and check if girth decreases? 
        # No, we want to INCREASE girth.
        
        # Heuristic: 
        # 1. Randomly pick 2 edges (u1, v1) and (u2, v2).
        # 2. Ensure no loops/multi-edges after swap.
        # 3. Perform swap.
        # 4. Check if girth < g. 
        #    If yes, rollback.
        #    If no, keep it? 
        #    But how do we know it IMPROVED? 
        #    We just need to eventually reach no cycle < g.
        
        # This is a random walk. With N=936, it might take a while.
        # But we have 10s budget per N.
        
        # Better: Targeted Swaps.
        # If we find a cycle, try to swap edges ON that cycle.
        
        # Let's assume the reconstruction of cycle edges is too slow/buggy.
        # Let's try "Global Random Swaps with Rollback".
        
        if random.random() < 0.5:
             # Random swap
             e1_idx = random.randrange(N)
             e2_idx = random.randrange(N)
             
             u1, v1 = adj[e1_idx][0], adj[e1_idx][1]
             u2, v2 = adj[e2_idx][0], adj[e2_idx][1]
             
             # Swap (u1, v1) and (u2, v2) to (u1, u2) and (v1, v2)? 
             # No, 3-regular swap:
             # Edges (a,b) and (c,d).
             # New edges (a,c) and (b,d).
             # Must ensure a!=c, b!=d, a!=d, b!=c (no loops).
             # And no multi-edges.
             
             # Pick 2 random edges from the graph
             # Need to store edges in a list for easy access
             
             pass
             
    # If loop ends, return None
    return None, None

print("Constructed")
