import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Vertices 0..15
    n = 16
    vertices = list(range(n))
    
    # Initialize with K3,13
    # Part A: 0, 1, 2
    # Part B: 3..15
    part_a = [0, 1, 2]
    part_b = list(range(3, 16))
    
    edges = set()
    for a in part_a:
        for b in part_b:
            edges.add((a, b))
            edges.add((b, a))
            
    # Adjacency lists for fast lookup
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # List of non-edges to try adding
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [p for p in all_pairs if p not in edges]
    random.shuffle(non_edges)
    
    def has_q3_using(u, v):
        # Check if adding edge (u,v) creates a Q3
        # u and v are now adjacent.
        # We look for the specific structure derived above.
        
        # Candidates for u1, u2 (neighbors of u in G, excluding v)
        cu = adj[u] - {v}
        cv = adj[v] - {u}
        
        if len(cu) < 2 or len(cv) < 2:
            return False
            
        # Iterate over pairs for u
        for u1, u2 in itertools.combinations(cu, 2):
            # Iterate over pairs for v
            for v1, v2 in itertools.combinations(cv, 2):
                # Check edges u1-v1 and u2-v2
                # Note: u1, u2 are distinct, v1, v2 are distinct.
                # u1, u2 != v1, v2? 
                # In the cube, u1 (010) and v1 (011) are distinct.
                # Can u1 == v1? No, u1 is neighbor of u, v1 is neighbor of v.
                # In cube, u=000, v=001. u1=010, v1=011. Distinct.
                # However, in the graph, a vertex could be both?
                # If u1 == v1, then that vertex is neighbor of both u and v.
                # In the cube structure I defined, u1 and v1 are distinct.
                # But there are other isomorphisms.
                # The specific structure:
                # u(000)-v(001)
                # u1(010)-v1(011)
                # u2(100)-v2(101)
                # w1(110)-w2(111)
                # Edges: u1-v1, u2-v2.
                # If u1 == v1, then edge (u1, v1) is a self-loop, impossible.
                # So we must ensure u1 != v1 and u2 != v2.
                # Also u1 != v2 and u2 != v1?
                # In this specific labeling, u1(010) and v2(101) are distinct.
                # So we require {u1, u2} and {v1, v2} to be disjoint?
                # Not necessarily disjoint sets, but the specific pairs must be valid edges.
                # If u1 == v1, the edge (u1, v1) doesn't exist (simple graph).
                # So if u1 == v1, skip.
                
                if u1 == v1 or u2 == v2:
                    continue
                    
                if v1 not in adj[u1] or v2 not in adj[u2]:
                    continue
                    
                # Now find w1, w2
                # w1 is common neighbor of u1, u2
                # w2 is common neighbor of v1, v2
                # w1, w2 must be distinct from u, v, u1, u2, v1, v2
                used = {u, v, u1, u2, v1, v2}
                
                cand_w1 = (adj[u1] & adj[u2]) - used
                cand_w2 = (adj[v1] & adj[v2]) - used
                
                if not cand_w1 or not cand_w2:
                    continue
                    
                for w1 in cand_w1:
                    for w2 in cand_w2:
                        if w1 == w2:
                            continue
                        if w2 in adj[w1]:
                            return True
                            
        return False

    best_edges = len(edges) // 2
    best_edge_list = list(edges)
    
    # Local search
    for u, v in non_edges:
        if time.time() > deadline - 5.0:
            break
            
        # Tentatively add
        edges.add((u, v))
        edges.add((v, u))
        adj[u].add(v)
        adj[v].add(u)
        
        if not has_q3_using(u, v):
            best_edges = len(edges) // 2
            best_edge_list = list(edges)
            # Keep the edge
        else:
            # Revert
            edges.remove((u, v))
            edges.remove((v, u))
            adj[u].remove(v)
            adj[v].remove(u)
            
    # Return list of [u, v]
    # Ensure unique undirected edges
    final_edges = []
    seen = set()
    for u, v in best_edge_list:
        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        final_edges.append([u, v])
        
    return final_edges
