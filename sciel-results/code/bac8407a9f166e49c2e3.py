def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s kill)
    
    def check_c4_free(n, adj):
        """Check if graph is C4-free (no two vertices have >1 common neighbor)."""
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = len(adj[u] & adj[v])
                    if common > 1:
                        return False
        return True
    
    def find_max_independent_set(n, adj, limit=11):
        """Find maximum independent set size. Returns size if < limit, else limit."""
        # Use a simple branch and bound
        # Greedy lower bound first
        def greedy_is():
            remaining = set(range(n))
            is_set = []
            while remaining:
                # Pick vertex with min degree in remaining
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                is_set.append(min_v)
                remaining.discard(min_v)
                remaining -= adj[min_v]
            return len(is_set)
        
        # If greedy gives >= limit, we might have an independent set of that size
        # But greedy isn't exact. Let's use a more exact method for small n.
        
        # Exact branch and bound
        # We want to know if alpha >= limit.
        # If we can find an independent set of size 'limit', return limit (or > limit).
        # If we prove max < limit, return the max.
        
        best = 0
        # Simple backtracking
        # To speed up, we can use bitmasks if n <= 64, but n can be up to 43.
        # Let's use sets for simplicity, but it might be slow.
        # Optimized:
        
        def backtrack(current_is, candidates):
            nonlocal best
            if len(current_is) > best:
                best = len(current_is)
                if best >= limit:
                    return
            
            if not candidates:
                return
            
            # Upper bound: len(current_is) + len(candidates)
            if len(current_is) + len(candidates) <= best:
                return
            
            # Choose a vertex from candidates
            # Heuristic: pick vertex with max degree in candidates to reduce branching
            # Or min degree?
            # Let's pick one.
            v = next(iter(candidates))
            
            # Branch 1: Include v
            new_candidates = candidates - {v} - adj[v]
            backtrack(current_is | {v}, new_candidates)
            
            # Branch 2: Exclude v
            new_candidates = candidates - {v}
            backtrack(current_is, new_candidates)
            
        backtrack(set(), set(range(n)))
        return best

    def try_construct_random(n, seed):
        """Try to construct a C4-free graph on n vertices with alpha < 11."""
        random.seed(seed)
        
        # Start with empty graph
        adj = [set() for _ in range(n)]
        
        # We want to add edges to reduce alpha.
        # Start with a random permutation of possible edges.
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u,v) creates C4
            # C4 is created if u and v have >1 common neighbor.
            # Currently, common neighbors of u and v are adj[u] & adj[v].
            # If we add (u,v), it doesn't change common neighbors of u and v directly,
            # but it adds v to adj[u] and u to adj[v].
            # Wait, C4 condition: no two vertices have >1 common neighbor.
            # Adding edge (u,v) makes u a neighbor of v and v a neighbor of u.
            # This increases common neighbors for pairs involving u or v?
            # No, common neighbors of x,y is N(x) ∩ N(y).
            # Adding (u,v) adds v to N(u) and u to N(v).
            # So for any x, N(x) ∩ N(u) might increase if x is neighbor of v?
            # Actually, we just need to check if adding (u,v) creates a C4.
            # A C4 is u-w-x-v-u.
            # This exists if there is a path u-w-x-v of length 3.
            # Equivalent: u and v have a common neighbor w, and w and x... no.
            # C4-free means: for any u,v, |N(u) ∩ N(v)| <= 1.
            # Before adding (u,v), let C = N(u) ∩ N(v).
            # If |C| >= 1, then adding (u,v) does NOT create a C4 involving u,v as opposite vertices?
            # Wait. If u and v are adjacent, they are not opposite in a C4.
            # In a C4, opposite vertices are not adjacent.
            # So if we add (u,v), u and v become adjacent.
            # The C4 condition applies to ALL pairs.
            # If u and v were not adjacent, and had 2 common neighbors w1, w2, then u-w1-v-w2-u is a C4.
            # So if |N(u) ∩ N(v)| >= 2, we CANNOT add (u,v) because it would mean u and v are in a C4?
            # No. If u and v are NOT adjacent, and have 2 common neighbors, the graph already has a C4.
            # So we must maintain |N(u) ∩ N(v)| <= 1 for ALL u,v.
            # If we add (u,v), we must ensure that for all x, |N(x) ∩ N(u)| <= 1 and |N(x) ∩ N(v)| <= 1.
            # Since u is added to N(v) and v to N(u):
            # For any x:
            # New N(x) ∩ N(u) = (Old N(x) ∩ N(u)) ∪ ({v} if v in N(x) else empty).
            # So if v in N(x), and Old N(x) ∩ N(u) already has 1 element, then New size is 2 -> C4.
            # Similarly for u in N(x).
            
            # So condition to add (u,v):
            # 1. u, v not already adjacent.
            # 2. For all x in N(u): v not in N(x) OR (v in N(x) and |N(x) ∩ N(u)| == 0).
            #    Wait, if v in N(x), then v is in New N(x) ∩ N(u).
            #    We need |New N(x) ∩ N(u)| <= 1.
            #    New N(x) ∩ N(u) = (Old N(x) ∩ N(u)) ∪ ({v} if v in N(x)).
            #    If v in N(x), then we need |Old N(x) ∩ N(u)| == 0.
            #    If v not in N(x), then we need |Old N(x) ∩ N(u)| <= 1 (which is already true).
            # 3. Similarly for all x in N(v): u not in N(x) OR (u in N(x) and |N(x) ∩ N(v)| == 0).
            
            # This check is O(deg(u) + deg(v)).
            
            # Optimization:
            # Precompute common neighbors?
            # Let's just do the check.
            
            can_add = True
            # Check for x in N(u)
            for x in adj[u]:
                if v in adj[x]:
                    # v is in N(x).
                    # Check if x and u have other common neighbors.
                    # If yes, adding (u,v) creates C4 x-u-v-... wait.
                    # If x in N(u) and x in N(v), then x is a common neighbor of u and v.
                    # If we add (u,v), do we create a C4?
                    # The C4 would be x-u-?-v-x.
                    # No, x is already connected to u and v.
                    # If we add (u,v), we have triangle x-u-v-x.
                    # Triangles are allowed.
                    # The constraint is on COMMON NEIGHBORS.
                    # For pair (x, u):
                    # N(x) contains v (new). N(u) contains x (old).
                    # Common neighbors of x and u:
                    # Old: N(x) ∩ N(u).
                    # New: (N(x) ∪ {v}) ∩ (N(u) ∪ {x}) ... wait N(u) doesn't change for x?
                    # N(u) gains v. N(x) gains u.
                    # Common neighbors of x and u in new graph:
                    # (Old N(x) ∩ Old N(u)) ∪ ({v} if v in Old N(x)) ∪ ({u} if u in Old N(x)) ...
                    # u is not in N(x) usually? No, u in N(x) because x in N(u).
                    # So u is a common neighbor? No, u is not a neighbor of u.
                    # Let's be precise.
                    # N_new(u) = N_old(u) ∪ {v}
                    # N_new(v) = N_old(v) ∪ {u}
                    # For any pair (a,b), N_new(a) ∩ N_new(b) must have size <= 1.
                    
                    # Case a=x, b=u.
                    # N_new(x) = N_old(x) ∪ {u} (if x in N_old(v)? No, x in N_old(u). Does x gain v? No, only u and v gain each other).
                    # Wait, only N(u) and N(v) change.
                    # N_new(x) = N_old(x) for x not in {u,v}.
                    # So for x not in {u,v}:
                    # N_new(x) ∩ N_new(u) = N_old(x) ∩ (N_old(u) ∪ {v}) = (N_old(x) ∩ N_old(u)) ∪ (N_old(x) ∩ {v}).
                    # We know |N_old(x) ∩ N_old(u)| <= 1.
                    # If v in N_old(x), then we add v to the intersection.
                    # So if v in N_old(x) AND |N_old(x) ∩ N_old(u)| == 1, then size becomes 2. C4!
                    # So condition: For all x in N_old(u), if v in N_old(x), then |N_old(x) ∩ N_old(u)| must be 0.
                    
                    if len(adj[x] & adj[u]) > 0:
                        can_add = False
                        break
            
            if not can_add:
                continue
                
            # Check for x in N(v)
            for x in adj[v]:
                if u in adj[x]:
                    if len(adj[x] & adj[v]) > 0:
                        can_add = False
                        break
            
            if can_add:
                adj[u].add(v)
                adj[v].add(u)
        
        # After greedy, check alpha
        # We can check alpha periodically or at the end.
        # Checking alpha is expensive.
        # Let's check if alpha < 11.
        
        # Heuristic: if degree is high, alpha is likely low.
        # Avg degree = 2E/n.
        # If avg degree > 5, alpha is likely < 11 for n=43?
        # Ramsey R(11, C4) ?
        
        # Let's just run the exact check.
        alpha = find_max_independent_set(n, adj, limit=11)
        
        if alpha < 11:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        return None

    # Strategy:
    # Try n from 43 down to 39.
    # For each n, try multiple random seeds.
    
    best_result = None
    
    for n in range(43, 38, -1):
        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            result = try_construct_random(n, seed)
            if result:
                best_result = result
                found = True
                break
        
        if found:
            break
            
    if best_result:
        return best_result
        
    # Fallback: Disjoint union of K3 (triangles)
    # n=39 -> 13 triangles.
    # Alpha of K3 is 1.
    # Alpha of 13 disjoint K3 is 13.
    # 13 >= 11. So this is INVALID for "no independent set of 11".
    # Wait, the task says "no independent set of 11 vertices".
    # So alpha must be < 11.
    # Disjoint K3s have alpha = n/3.
    # To have alpha < 11, we need n/3 < 11 => n < 33.
    # So n=30 (10 triangles) has alpha 10 < 11. Valid.
    
    # Can we do better?
    # Union of K4s?
    # K4 is C4-free? No, K4 has C4s. (1-2-3-4-1).
    # So K4 is not allowed.
    
    # What about K5? No.
    
    # We need C4-free graphs.
    # Complete graphs K3 are C4-free.
    # Petersen graph (10 vertices, 3-reg, C4-free, alpha 4).
    # Union of Petersen graphs?
    # 4 Petersen graphs = 40 vertices.
    # Alpha = 4 * 4 = 16. Too high.
    
    # We need to connect them or use a denser C4-free graph.
    
    # The random greedy approach above should find something if it exists.
    # If not, we return the best valid one.
    
    # Let's try n=32 with random greedy again, maybe more seeds.
    for n in range(32, 38, -1):
        if time.time() > deadline - 5:
            break
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            result = try_construct_random(n, seed)
            if result:
                return result
                
    # Final fallback: n=30 (10 triangles)
    n = 30
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
        
    return {'n': n, 'edges': edges}
