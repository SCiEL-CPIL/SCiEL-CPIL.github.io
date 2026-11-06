import numpy as np

def construct():
    # Parameters
    q = 7
    # Generate points of PG(2, q)
    # Points are non-zero vectors in GF(q)^3 up to scaling
    # We can represent them as tuples (x, y, z) with z=1 (affine part) or z=0 (line at infinity)
    # Total points: q^2 + q + 1
    
    points = []
    # Affine points: (x, y, 1)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (x, 1, 0) and (1, 0, 0)
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_total = len(points) # 57
    
    # Map points to indices 0..56
    point_index = {p: i for i, p in enumerate(points)}
    
    # Define polarity
    # Standard polarity for PG(2, q) with q odd:
    # u . v = u1*v1 + u2*v2 + u3*v3 = 0
    # Note: This form is symmetric.
    # Two points u, v are adjacent if dot(u, v) == 0
    
    adj = [set() for _ in range(n_total)]
    
    for i in range(n_total):
        u = points[i]
        for j in range(i + 1, n_total):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    # We want to find a subset R of size k such that 
    # for all u not in R, |N(u) intersection R| <= D
    # where D = 8 - (n_target - 39)
    # Base degree is 8.
    
    def try_find_subset(n_target):
        k = n_total - n_target
        min_deg_req = n_target - 39
        D = 8 - min_deg_req
        
        if D < 0:
            return None
            
        # Greedy with backtracking
        # We pick k vertices for R
        # State: R (list of indices), loss (array of length n_total)
        # loss[u] = number of neighbors of u already in R
        
        import random
        random.seed(42) # Deterministic for reproducibility, but we can try multiple seeds if needed
        
        def backtrack(R, loss, count):
            if count == k:
                return R
            
            # Candidates: vertices not in R
            # We want to pick v such that for all u in N(v) \ R, loss[u] < D
            # If we pick v, loss[u] increases by 1 for u in N(v).
            # So we need loss[u] + 1 <= D => loss[u] < D for all u in N(v) \ R.
            
            # To speed up, we can filter candidates
            candidates = []
            in_R = set(R)
            for v in range(n_total):
                if v in in_R:
                    continue
                valid = True
                for u in adj[v]:
                    if u in in_R:
                        continue
                    if loss[u] >= D:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if not candidates:
                return None
                
            # Shuffle candidates for variety
            random.shuffle(candidates)
            
            for v in candidates:
                # Try adding v
                new_R = R + [v]
                new_loss = loss.copy()
                for u in adj[v]:
                    new_loss[u] += 1
                    
                res = backtrack(new_R, new_loss, count + 1)
                if res is not None:
                    return res
            return None
            
        initial_loss = [0] * n_total
        # Start with empty R
        res = backtrack([], initial_loss, 0)
        
        if res is None:
            return None
            
        R = set(res)
        V_remaining = [i for i in range(n_total) if i not in R]
        
        # Verify
        for u in V_remaining:
            deg = len([v for v in adj[u] if v not in R])
            if deg < min_deg_req:
                return None # Should not happen if backtrack worked correctly
        
        return V_remaining, R

    # Try n=47, 46, 45... down to 43
    for n_target in [47, 46, 45, 44, 43]:
        res = try_find_subset(n_target)
        if res is not None:
            V_remaining, R = res
            n = len(V_remaining)
            
            # Remap vertices to 0..n-1
            old_to_new = {old: new for new, old in enumerate(V_remaining)}
            
            edges = []
            for u in V_remaining:
                for v in adj[u]:
                    if v in old_to_new:
                        if old_to_new[u] < old_to_new[v]:
                            edges.append([old_to_new[u], old_to_new[v]])
            
            return {'n': n, 'edges': edges}
            
    # Fallback: return the q=6 polarity graph if all else fails (should not happen)
    # But we can just return a valid smaller graph
    # For safety, return n=39 empty graph if something goes wrong
    return {'n': 39, 'edges': []}
