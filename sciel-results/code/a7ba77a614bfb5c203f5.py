import time
import itertools

def construct():
    start_time = time.time()
    n = 16
    vertices = list(range(n))
    
    # Precompute all edges
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
            
    # Adjacency list for fast neighbor lookup
    # We will maintain this dynamically or just use a set of edges.
    # For speed, let's maintain a set of edges `current_edges` and a list of neighbors `neighbors`.
    
    # Initial state: Empty graph? Or a greedy start?
    # Greedy start: Add edges in a random order or specific order.
    # Let's start with a known good structure or just empty.
    # Empty is safe but slow to build up.
    # K_{3,13} has 39 edges.
    # Let's start with K_{3,13}.
    current_edges = set()
    for a in range(3):
        for b in range(3, 16):
            current_edges.add((a, b) if a < b else (b, a))
            
    neighbors = {i: set() for i in range(n)}
    for u, v in current_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    best_count = len(current_edges)
    best_edges = set(current_edges)
    
    # Candidates: edges not in current_edges
    candidates = list(all_edges - current_edges)
    
    # Sort candidates to find good solutions earlier?
    # Maybe edges within the larger part B (3..15) first?
    def edge_key(e):
        u, v = e
        if u >= 3 and v >= 3:
            return 0
        elif u < 3 and v < 3:
            return 1
        else:
            return 2
    candidates.sort(key=edge_key)
    
    def creates_q3(u, v, current_edges, neighbors):
        """
        Check if adding edge (u,v) to the graph creates a Q3.
        Assumes (u,v) is not in current_edges.
        """
        # In Q3, u and v are adjacent.
        # u has 2 other neighbors in Q3: x3, x4
        # v has 2 other neighbors in Q3: x2, x5
        # x3, x4 are in neighbors[u] (excluding v, which is not there yet)
        # x2, x5 are in neighbors[v] (excluding u)
        
        n_u = neighbors[u]
        n_v = neighbors[v]
        
        # We need to pick x3, x4 from n_u and x2, x5 from n_v
        # such that:
        # 1. x2, x3, x4, x5 are distinct
        # 2. (x2, x3) is in current_edges
        # 3. (x4, x5) is in current_edges
        # 4. There exist x6, x7 such that:
        #    (x2, x6) in E
        #    (x3, x7) in E
        #    (x5, x6) in E
        #    (x6, x7) in E
        #    (x7, x4) in E
        
        # To optimize, iterate over possible (x2, x3) pairs first?
        # Or iterate over x3 in n_u, x4 in n_u, x2 in n_v, x5 in n_v?
        # That's O(deg^4).
        
        # Better:
        # Iterate over x3 in n_u
        # Iterate over x4 in n_u, x4 != x3
        # Iterate over x2 in n_v
        # If (x2, x3) not in E, continue
        # Iterate over x5 in n_v
        # If (x4, x5) not in E, continue
        # If x2, x3, x4, x5 not distinct, continue
        # Now check for x6, x7.
        # x6 must be neighbor of x2 and x5.
        # x7 must be neighbor of x3 and x4 and x6.
        # Also (x6, x7) must be in E.
        
        n_u_list = list(n_u)
        n_v_list = list(n_v)
        
        for x3 in n_u_list:
            for x4 in n_u_list:
                if x4 == x3: continue
                for x2 in n_v_list:
                    if x2 == x3 or x2 == x4: continue
                    # Check (x2, x3) in E
                    e23 = (x2, x3) if x2 < x3 else (x3, x2)
                    if e23 not in current_edges: continue
                    
                    for x5 in n_v_list:
                        if x5 == x2 or x5 == x3 or x5 == x4: continue
                        # Check (x4, x5) in E
                        e45 = (x4, x5) if x4 < x5 else (x5, x4)
                        if e45 not in current_edges: continue
                        
                        # Found x2, x3, x4, x5
                        # Now find x6, x7
                        # x6 in neighbors[x2] AND neighbors[x5]
                        common_25 = neighbors[x2] & neighbors[x5]
                        # x7 in neighbors[x3] AND neighbors[x4]
                        common_34 = neighbors[x3] & neighbors[x4]
                        
                        # We need x6 in common_25, x7 in common_34
                        # such that (x6, x7) in E
                        # And x6, x7 distinct from x2,x3,x4,x5 and u,v
                        # (They are distinct because x6 is neighbor of x2, and x2 is not u/v? 
                        # Wait, x2 is neighbor of v. x6 is neighbor of x2.
                        # x6 could be u or v?
                        # If x6 == u: (x2, u) in E? x2 is neighbor of v. Is x2 neighbor of u?
                        # In Q3, x2 (vertex 2) is not neighbor of u (vertex 0).
                        # So if (x2, u) in E, it's not part of this Q3 embedding.
                        # But we are checking if the *graph* contains Q3.
                        # The vertices of the Q3 must be distinct.
                        # So x6 != u, x6 != v, x6 != x2, x6 != x3, x6 != x4, x6 != x5.
                        # x7 != u, x7 != v, x7 != x2, x7 != x3, x7 != x4, x7 != x5.
                        # Also x6 != x7.
                        
                        forbidden = {u, v, x2, x3, x4, x5}
                        
                        for x6 in common_25:
                            if x6 in forbidden: continue
                            # x7 must be in common_34 AND neighbors[x6]
                            # And x7 not in forbidden
                            # And x7 != x6
                            for x7 in common_34:
                                if x7 in forbidden: continue
                                if x7 == x6: continue
                                e67 = (x6, x7) if x6 < x7 else (x7, x6)
                                if e67 in current_edges:
                                    return True
        return False

    def search(current_edges, neighbors, candidates, count):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > 110:
            return
            
        # Upper bound: count + len(candidates)
        if count + len(candidates) <= best_count:
            return
            
        if not candidates:
            if count > best_count:
                best_count = count
                best_edges = set(current_edges)
            return
            
        # Pick a candidate
        # Heuristic: pick edge with lowest degree sum? Or just first?
        e = candidates[0]
        rest = candidates[1:]
        u, v = e
        
        # Try adding e
        if not creates_q3(u, v, current_edges, neighbors):
            new_edges = set(current_edges)
            new_edges.add(e)
            new_neighbors = {i: set(neighbors[i]) for i in range(n)}
            new_neighbors[u].add(v)
            new_neighbors[v].add(u)
            
            # Remove e from candidates? No, candidates is a list.
            # We just pass rest.
            # But we should remove e from any future candidate lists?
            # Since we are branching, we just use 'rest' which excludes e.
            
            search(new_edges, new_neighbors, rest, count + 1)
            
        # Try skipping e
        search(current_edges, neighbors, rest, count)
        
    search(current_edges, neighbors, candidates, best_count)
    
    return [list(e) for e in best_edges]
