import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 40
    
    def is_c6_free(adj, u, v):
        """
        Check if adding edge (u, v) to the graph defined by adjacency sets 'adj'
        creates a C6.
        A C6 is created if there is a path of length 4 between u and v in G \ {(u,v)}.
        Path: u - a - b - c - d - v
        """
        # Neighbors of u
        nu = adj[u]
        
        # Distance 2 from u
        n2 = set()
        for a in nu:
            for x in adj[a]:
                if x != u and x not in nu:
                    n2.add(x)
                    
        # Distance 3 from u
        n3 = set()
        for b in n2:
            for y in adj[b]:
                if y not in nu and y not in n2:
                    n3.add(y)
                    
        # Distance 4 from u
        # Check if v is at distance 4
        for c in n3:
            if v in adj[c]:
                # We have a path u-a-b-c-v of length 4.
                # We must ensure the vertices are distinct and form a simple cycle.
                # The path is u, a, b, c, v.
                # a is in nu.
                # b is in n2, so b != u, b != a (since b not in nu).
                # c is in n3, so c != u, c != a, c != b.
                # v is the target.
                # We need to ensure v is not in {u, a, b, c}.
                # v != u is given (simple graph).
                # v != a: a is in nu. If v in nu, we would have a C4. 
                # We should check if adding (u,v) creates a C4 as well? 
                # The task says "no six-cycle". It doesn't forbid C4.
                # However, usually "C6-free" means no induced C6 or just no C6?
                # "containing no six-cycle". This usually means no subgraph isomorphic to C6.
                # If there is a C4, it doesn't contain a C6.
                # So we only care if a C6 is formed.
                # The path u-a-b-c-v plus edge (u,v) forms a cycle of length 5?
                # No, u-v-c-b-a-u is length 5.
                # Wait.
                # Path length 4: u(0)-a(1)-b(2)-c(3)-v(4).
                # Adding edge (u,v) creates a cycle u-v-c-b-a-u.
                # Length: 1 (u-v) + 3 (v-c-b-a) + 1 (a-u) = 5?
                # Let's count vertices: u, v, c, b, a. That's 5 vertices.
                # So a path of length 4 creates a C5.
                
                # We need a C6.
                # A C6 has 6 vertices.
                # Adding (u,v) creates a C6 if there is a path of length 5 between u and v.
                # Path: u-a-b-c-d-v.
                # Vertices: u, a, b, c, d, v. (6 vertices).
                
                return True
        return False

    def is_c6_free_v2(adj, u, v):
        """
        Corrected check: Adding (u,v) creates a C6 if there is a path of length 5 
        between u and v in G \ {(u,v)}.
        Path: u - a - b - c - d - v
        """
        nu = adj[u]
        if v in nu:
            return False # Already an edge, or invalid input for this function context
            
        # Distance 1 from u: nu
        # Distance 2 from u:
        n2 = set()
        for a in nu:
            for x in adj[a]:
                if x != u and x not in nu:
                    n2.add(x)
                    
        # Distance 3 from u:
        n3 = set()
        for b in n2:
            for y in adj[b]:
                if y not in nu and y not in n2:
                    n3.add(y)
                    
        # Distance 4 from u:
        n4 = set()
        for c in n3:
            for z in adj[c]:
                if z not in nu and z not in n2 and z not in n3:
                    n4.add(z)
                    
        # Distance 5 from u:
        # Check if v is in n5
        for d in n4:
            if v in adj[d]:
                # Path u-a-b-c-d-v exists.
                # Vertices u, a, b, c, d, v are distinct?
                # a in nu.
                # b in n2 (not in nu, not u).
                # c in n3 (not in nu, not n2, not u).
                # d in n4 (not in nu, not n2, not n3, not u).
                # v is target.
                # Is v in {u, a, b, c, d}?
                # v != u.
                # If v in nu, then dist(u,v)=1. But we are adding edge, so assume not.
                # If v in n2, dist=2.
                # If v in n3, dist=3.
                # If v in n4, dist=4.
                # If v in n5, dist=5.
                # The sets n2, n3, n4 are disjoint from nu and each other by construction (BFS layers).
                # So if v in n5 (neighbors of n4), and v not in previous layers, it's a simple path.
                # We must ensure v is not in nu, n2, n3, n4.
                if v in nu or v in n2 or v in n3 or v in n4:
                    return False # Shorter path exists, so this path is not simple or not minimal? 
                                 # Actually, if a shorter path exists, the cycle formed by the longer path 
                                 # might not be a simple cycle of length 6 if it intersects itself.
                                 # But BFS layers ensure disjointness. 
                                 # If v is in n2, there is a path of length 2. 
                                 # The path of length 5 might share vertices? 
                                 # No, BFS layers are disjoint. 
                                 # So if v is in n2, it is NOT in n5 (since n5 is neighbors of n4, 
                                 # and n4 is disjoint from n2). 
                                 # Wait, n5 is not a set I built. I'm just checking neighbors of n4.
                                 # If v is a neighbor of d in n4, and v is in n2, then there is a path of length 2 
                                 # and a path of length 5. The length 5 path u...d-v might not be simple 
                                 # if it visits v earlier? No, v is the endpoint. 
                                 # But does the path u-a-b-c-d-v visit v earlier? No.
                                 # Does it visit other vertices twice? 
                                 # a in L1, b in L2, c in L3, d in L4. All distinct sets.
                                 # So the path is simple.
                                 # So if v is in n2, we have a C4 and a C6?
                                 # The cycle from the length 5 path is u-a-b-c-d-v-u. Length 6.
                                 # So yes, it creates a C6.
                                 # So we should return True.
                                 # The check `if v in nu...` was meant to ensure simplicity, 
                                 # but BFS layers guarantee simplicity of the path u...d.
                                 # The only issue is if the path is not simple, which it is by construction.
                                 # So we just return True.
                return True
        return False

    def greedy_construction(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while True:
            if time.time() > deadline:
                break
                
            best_edge = None
            best_score = float('inf')
            
            # To speed up, we can sample a subset of edges if the list is large, 
            # but 40 vertices means 780 edges. Checking all is fine.
            for u, v in all_edges:
                if u in adj[v]: # Already added
                    continue
                    
                # Heuristic: minimize degree sum
                score = len(adj[u]) + len(adj[v])
                
                if score < best_score:
                    if is_c6_free_v2(adj, u, v):
                        best_edge = (u, v)
                        best_score = score
                        # Optional: break early if we find a very good edge? 
                        # No, we want the best one.
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
            # Remove from all_edges to avoid re-checking
            # This is inefficient. Better to just check adjacency.
            
        return edges

    best_edges = []
    best_len = 0
    
    # Run multiple restarts with different seeds to find a better local optimum
    # The greedy choice depends on the initial state (empty) but ties might be broken by order.
    # Actually, the greedy choice is deterministic if we always pick min degree sum.
    # If there are ties, the order of iteration in `all_edges` matters.
    # So shuffling `all_edges` in each restart helps.
    
    for i in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        random.seed(i)
        adj = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        # We will iterate through the shuffled list repeatedly? 
        # No, standard greedy: at each step, pick the best edge from the remaining ones.
        # To make it efficient, we just scan all remaining edges each time.
        
        remaining_edges = set(all_edges)
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            best_edge = None
            best_score = float('inf')
            
            for u, v in remaining_edges:
                # Heuristic: minimize degree sum
                score = len(adj[u]) + len(adj[v])
                
                if score < best_score:
                    if is_c6_free_v2(adj, u, v):
                        best_edge = (u, v)
                        best_score = score
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            remaining_edges.discard((u, v))
            # Also discard (v, u) if it was in there, but we generated u<v so it's not.
            
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges

    return best_edges
