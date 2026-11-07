"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    # The seed program returns a star graph with 15 edges.
    # We aim to find a denser graph that is edge-critical for diameter 2.
    # Strategy: Start with the complete graph K_n and remove edges such that:
    # 1. The resulting graph G has diameter <= 2.
    # 2. G is edge-critical for diameter 2 (removing any edge increases diameter or disconnects).
    #
    # Insight: If we let H be the complement of G, then:
    # - G has diameter <= 2 iff for every pair u,v not adjacent in G (i.e., (u,v) in H), there is a path of length 2 in G.
    # - G is edge-critical for diameter 2 iff for every edge (u,v) in G, either:
    #   a) Removing (u,v) disconnects G (u and v are in different components after removal), OR
    #   b) Removing (u,v) makes the distance between u and v greater than 2.
    #
    # A known construction for such graphs is the complement of a 2-regular graph (a union of cycles).
    # If H is a cycle C_n, then G = K_n \ C_n.
    # Let's verify if G = K_16 \ C_16 is edge-critical for diameter 2.
    #
    # In G = K_n \ C_n, the missing edges are (i, i+1) mod n.
    # For any edge (u, v) in G, u and v are not consecutive in the cycle.
    # If we remove (u, v), is the distance between u and v > 2?
    # u and v are connected by a path of length 2 in G if there exists w such that (u,w) in G and (w,v) in G.
    # (u,w) not in G iff w = u+1 or w = u-1 (mod n).
    # (w,v) not in G iff w = v+1 or w = v-1 (mod n).
    # So a common neighbor w exists unless {u+1, u-1} intersects {v+1, v-1} in a way that blocks all w.
    # Actually, we need to check if there is ANY w such that w is not in {u+1, u-1, v+1, v-1}.
    # Since n=16, there are 16 vertices. The set of forbidden w for a path u-w-v has size at most 4.
    # So there are at least 12 valid w's. Thus, removing any edge (u,v) in G leaves a path of length 2 between u and v.
    # Wait, this means the distance remains 2. So the diameter remains 2.
    # But the condition for edge-criticality is that removing the edge INCREASES the diameter above 2 OR disconnects.
    # If the distance between u and v remains 2, but the diameter of the whole graph might increase?
    # No, if the distance between every pair remains <= 2, the diameter is still 2.
    # So G = K_n \ C_n is NOT edge-critical for diameter 2, because removing any edge still leaves diameter 2.
    #
    # We need a graph where removing ANY edge breaks the diameter 2 property.
    # This implies that for every edge (u,v), the only path of length <= 2 between u and v is the edge itself? No, that's impossible for diameter 2 graphs with n>3.
    # It implies that for every edge (u,v), there is NO path of length 2 between u and v in G \ {(u,v)}.
    # This means that for every edge (u,v) in G, the sets N(u) \ {v} and N(v) \ {u} are disjoint.
    # In other words, u and v have no common neighbors.
    #
    # If u and v have no common neighbors, then in the complement H, every other vertex w is non-adjacent to u OR non-adjacent to v in G.
    # i.e., for every w != u,v, (u,w) in H or (v,w) in H.
    # This is a strong condition.
    #
    # Let's try a different approach: Local search.
    # Start with a random graph with high edge density.
    # Check if it is edge-critical for diameter 2.
    # If not, identify an edge whose removal does not increase diameter, and remove it? No, we want to MAXIMIZE edges.
    # So we should ADD edges if possible, or REMOVE edges that are "redundant" for diameter 2.
    # An edge (u,v) is redundant if G \ {(u,v)} still has diameter <= 2.
    # We want a graph with NO redundant edges.
    #
    # Algorithm:
    # 1. Start with K_n.
    # 2. While there is an edge (u,v) such that G \ {(u,v)} has diameter <= 2:
    #    - Remove (u,v).
    # 3. This yields a maximal edge-critical graph for diameter 2.
    # 4. To maximize edges, we should remove as few edges as possible.
    #    The order of removal matters. We want to remove edges that are "most redundant" or in a way that preserves connectivity and diameter 2 for as long as possible.
    #    Actually, we just want to remove edges until no more are redundant.
    #    This process is deterministic if we fix the order of edges.
    #    We can try different orders or random restarts to find a graph with more edges.
    #
    # Let's implement this greedy removal.
    start_time = time.time()
    deadline = start_time + 45.0  # Leave some margin for the 60s limit
    
    best_edges = None
    best_count = -1
    
    # Try multiple random orderings of edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        G = set(all_edges)
        
        # Helper to check diameter
        def has_diameter_2(edge_set):
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check all pairs
            for u in range(n):
                # BFS from u
                dist = [-1] * n
                dist[u] = 0
                queue = [u]
                for x in queue:
                    if dist[x] >= 2:
                        continue
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            if dist[y] <= 2:
                                queue.append(y)
                if any(d == -1 for d in dist):
                    return False
                if any(d > 2 for d in dist):
                    return False
            return True
        
        # Greedy removal
        # We iterate through edges in the shuffled order.
        # If removing the edge keeps diameter <= 2, we remove it.
        # This might be too aggressive. We want to remove ONLY if it is redundant.
        # But if we remove it, we might make other edges redundant.
        # This is a standard greedy approach for maximal independent sets, etc.
        
        # Optimization: We don't need to check diameter from scratch every time.
        # But for n=16, it's fast enough.
        
        current_edges = list(G)
        # We will iterate through the shuffled list of edges.
        # If an edge is still in G, we check if it's redundant.
        for u, v in all_edges:
            if (u, v) not in G and (v, u) not in G:
                continue
            if (u, v) in G:
                G.remove((u, v))
            else:
                G.remove((v, u))
                
            if has_diameter_2(G):
                # Keep it removed
                pass
            else:
                # Restore it
                G.add((u, v))
                
        if len(G) > best_count:
            best_count = len(G)
            best_edges = list(G)
            
        # Early exit if we reach the theoretical max?
        # Max edges for such a graph is unknown, but likely close to K_n.
        # If we find a graph with 100+ edges, it's very good.
        
    # Convert to required format
    if best_edges is None:
        return [[0, v] for v in range(1, n)]
        
    result = []
    for u, v in best_edges:
        result.append([u, v])
        
    return result
# EVOLVE-BLOCK-END
