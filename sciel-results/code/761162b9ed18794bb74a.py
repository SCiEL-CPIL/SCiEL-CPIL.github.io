"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build base adjacency and precompute distances for fast filtering
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # Precompute all-pairs shortest paths on the base graph (path)
    # Since it's a path, dist[u][v] = abs(u-v)
    # But we need to check if adding edges reduces diameter.
    
    # Candidate edges: only even-odd pairs not already in base
    # This ensures triangle-free property (bipartite graph)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 != 0:  # one even, one odd
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    candidates.append((u, v))
                    
    # Optimization: Pre-compute which candidate edges are "useful"
    # An edge (u,v) is useful if it creates a shortcut.
    # In a path, any non-adjacent edge is a shortcut.
    # Actually, we can just use all non-adjacent even-odd edges.
    
    # Fast diameter check using BFS on the augmented graph.
    # To speed up, we can check if the diameter is <= 4.
    # A graph has diameter <= 4 if for all pairs (u,v), dist(u,v) <= 4.
    # Equivalently, there exists a set of edges such that...
    # Let's use a BFS-based check.
    
    def compute_diameter(added_edges):
        # Build full adjacency
        full_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        for u, v in added_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
            
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                if dist[cur] >= 4: # Early exit if we've reached depth 4 and haven't found a larger one? No, we need to check all.
                    # Actually, if we reach depth 4, any further neighbor is depth 5, which is invalid.
                    # But we need to ensure connectivity too.
                    pass
                for nb in full_adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        if dist[nb] > 4:
                            return 5 # Invalid
                        queue.append(nb)
            for d in dist:
                if d == -1:
                    return float('inf') # Disconnected
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d

    # Lower bound estimation:
    # Each added edge can reduce the diameter.
    # In a path of 24 nodes, diameter is 23.
    # To get diameter 4, we need to "cover" the path.
    # A star centered at 12 has diameter 2 (if connected to all).
    # But we are constrained to even-odd edges.
    # Let's search by increasing number of edges.
    
    # Theoretical lower bound:
    # With k edges, max degree is k+1 (for hubs).
    # Diameter 4 implies that from any node, we can reach any other in 4 steps.
    # In a bipartite graph, distances are even? No, edges are even-odd, so path length 1 is E-O.
    # Distance 2 is E-E or O-O.
    # Distance 4 is E-E or O-O.
    # So all nodes must be within distance 4.
    
    # Let's start with a strong initial solution to have a "best" to beat.
    # The current best is 12.
    # Let's try to find a solution with fewer edges.
    
    # Heuristic: Hub-and-spoke is good.
    # Let's try 1 hub, then 2 hubs, etc.
    
    # Strategy: 
    # 1. Try to find a solution with k edges for k = 6, 7, 8...
    # 2. For each k, sample random subsets of candidates? No, too many.
    # 3. Use a constructive approach:
    #    - Pick a set of "hubs" (even nodes).
    #    - Connect odd nodes to nearest hub.
    #    - Check diameter.
    #    - If diameter > 4, add edges or change hubs.
    
    # Let's refine the hub search.
    # Number of hubs:
    # If we use 1 hub (even), it connects to all odd nodes.
    # Distances:
    # Odd-Odd: via hub (2 steps).
    # Even-Even: via odd neighbor (2 steps) or via hub?
    # If hub is 12:
    # Node 0 (even) -> 1 (odd) -> 12 (hub) -> 13 (odd) -> 14 (even). Dist 4.
    # Node 0 -> 1 -> 12 -> 11 -> 10. Dist 4.
    # Node 0 -> 1 -> 12 -> 23? No, 12-23 is not an edge unless added.
    # If hub 12 connects to all odds:
    # 0-1-12-23-22? No.
    # Let's check diameter for Hub=12, connected to all odds.
    # 0 to 23: 0-1-12-23. Dist 3.
    # 0 to 22: 0-1-12-23-22. Dist 4.
    # 1 to 22: 1-12-23-22. Dist 3.
    # 2 to 23: 2-1-12-23. Dist 3.
    # It seems 1 hub (12) connected to all 12 odd nodes works?
    # Wait, 12 is even. Odds are 1,3..23.
    # Edges: (12,1), (12,3), ..., (12,23).
    # Count: 12 edges.
    # This is the current best.
    
    # Can we do better?
    # If we use 2 hubs, say 6 and 18.
    # 6 connects to odds near 6. 18 connects to odds near 18.
    # What about middle odds?
    # If we split odds between two hubs, we save edges? No, we still need to cover all odds.
    # Each odd node MUST have a path to the rest.
    # In a bipartite graph, if an odd node has degree 1 (only connected to one even),
    # and that even is a hub, it's fine.
    # But if we don't connect an odd node directly to a hub, it must use path edges.
    # Path edges are fixed.
    # So an odd node v can be reached from hub H if there is a path v-...-H.
    # If we don't add edge (H,v), we rely on path.
    # Path distance from H to v is |H-v|.
    # If |H-v| is large, diameter might exceed 4.
    
    # Let's try to minimize edges by NOT connecting every odd to a hub.
    # Only connect an odd node v to a hub H if |H-v| > 2?
    # If |H-v| <= 2, the path distance is small, so maybe we don't need the shortcut.
    
    # Let's test a "Sparse Hub" strategy.
    # Hubs: even nodes.
    # For each odd node v, find if it's within distance 4 of all other nodes using current edges.
    
    # Let's run a search for k < 12.
    
    best_edges = None
    best_count = 12 # Current known best
    
    # Try k from 6 to 11
    for k in range(6, 12):
        # Randomized search with pruning
        # Or systematic:
        # Pick k candidate edges.
        # This is C(~100, 6) which is huge.
        
        # Better: Constructive search.
        # 1. Select a set of hubs.
        # 2. Assign odd nodes to hubs or leave them on path.
        
        # Let's try all pairs of hubs (h1, h2) where h1, h2 are even.
        # And single hubs.
        
        # Single hub
        for h in range(0, n, 2):
            # Connect odd nodes v to h if necessary.
            # "Necessary" means if we don't connect, diameter > 4.
            # Let's just connect all odds to h and then remove edges one by one.
            trial = [(h, v) for v in range(1, n, 2)]
            # Remove edges that are "redundant" for diameter 4.
            # Redundant if removing them keeps diameter <= 4.
            # Start with full star.
            current = trial[:]
            improved = True
            while improved:
                improved = False
                for i in range(len(current)):
                    # Skip if time is tight?
                    trial2 = current[:i] + current[i+1:]
                    if compute_diameter(trial2) <= 4:
                        current = trial2
                        improved = True
                        break
            if len(current) < best_count:
                best_count = len(current)
                best_edges = current[:]
                
        # Two hubs
        if time.time() > 85: break # Safety
        
        for h1 in range(0, n, 2):
            for h2 in range(h1+2, n, 2):
                # Assign each odd v to h1 or h2 or neither (path only).
                # Greedy assignment:
                # Start with no edges.
                # Check diameter.
                # If > 4, find a pair (u,v) with dist > 4.
                # Add an edge that reduces this distance.
                # This is a greedy set cover / hitting set approach.
                
                current = []
                # We will add edges until diameter <= 4.
                # To minimize edges, we want edges that cover multiple "long" pairs.
                
                # Let's use a different heuristic:
                # For each odd v, if dist_path(v, farthest_even) > 4, we need a shortcut.
                
                # Let's try a specific construction:
                # Connect v to h1 if |v-h1| < |v-h2|, else h2.
                # But only if necessary.
                
                # Let's just try the "nearest hub" assignment and then prune.
                trial = []
                for v in range(1, n, 2):
                    d1 = abs(v - h1)
                    d2 = abs(v - h2)
                    if d1 <= d2:
                        trial.append((h1, v))
                    else:
                        trial.append((h2, v))
                
                # Prune
                current = trial[:]
                improved = True
                while improved:
                    improved = False
                    for i in range(len(current)):
                        trial2 = current[:i] + current[i+1:]
                        if compute_diameter(trial2) <= 4:
                            current = trial2
                            improved = True
                            break
                if len(current) < best_count:
                    best_count = len(current)
                    best_edges = current[:]
                    
        # Three hubs?
        # Might be overkill.
        
    # If we found a better solution, return it.
    # Otherwise return the best found (which might be 12).
    
    if best_edges is None:
        # Fallback to the known 12-edge solution
        best_edges = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
        
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
# EVOLVE-BLOCK-END
