import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def build_polarity_graph(q):
        """Build polarity graph of projective plane of order q.
        Returns (n_vertices, adj_list) where adj_list[i] is set of neighbors."""
        # Projective plane PG(2,q) has q^2+q+1 points
        n_pts = q * q + q + 1
        
        # Enumerate all 1-dim subspaces of F_q^3
        points = []
        seen = set()
        for a in range(q):
            for b in range(q):
                # Point (a, b, 1)
                key = (a, b, 1)
                if key not in seen:
                    seen.add(key)
                    points.append((a, b, 1))
        # Points with z=0: (a, b, 0) where not both 0
        for a in range(q):
            for b in range(q):
                if a == 0 and b == 0:
                    continue
                key = (a, b, 0)
                if key not in seen:
                    seen.add(key)
                    points.append((a, b, 0))
        
        # We should have q^2+q+1 points
        # Define polarity: point p=(x,y,z) is incident to line L if x*L_x + y*L_y + z*L_z = 0 in F_q
        # For a self-polarity, we define a symmetric bilinear form
        # Use the standard polarity: (x1,y1,z1) ~ (x2,y2,z2) iff x1*y2 + y1*x2 + z1*z2 = 0 (mod q)
        # Actually, let's use a simpler approach: define the polarity matrix
        
        # Standard polarity for PG(2,q): 
        # p = (x,y,z) is on line l=(x',y',z') iff x*x' + y*y' + z*z' = 0
        # A point is self-incident if x^2 + y^2 + z^2 = 0
        # This gives a polarity (symmetric bilinear form)
        
        adj = [set() for _ in range(n_pts)]
        
        for i in range(n_pts):
            xi, yi, zi = points[i]
            for j in range(i + 1, n_pts):
                xj, yj, zj = points[j]
                # Check if point i is incident with line j (polarity)
                val = (xi * yj + yi * xj + zi * zj) % q
                if val == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return n_pts, adj
    
    def check_c4_safe(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4.
        A C4 would be u-x-y-v-u where x in N(u), y in N(v), x!=y, x~y."""
        nu = adj[u]
        nv = adj[v]
        # Need to find x in nu, y in nv, x != y, such that x ~ y
        if len(nu) > len(nv):
            nu, nv = nv, nu
        for x in nu:
            for y in nv:
                if x == y:
                    continue
                if y in adj[x]:
                    return True
        return False
    
    def greedy_build_from_seed(seed_n, seed_adj, target_n, min_deg_req, time_limit, rng):
        """Extend a C4-free seed graph to target_n vertices, maintaining C4-freeness
        and trying to achieve min degree >= min_deg_req."""
        t0 = time.time()
        
        # Start with the seed graph
        adj = [set() for _ in range(target_n)]
        for i in range(seed_n):
            adj[i] = set(seed_adj[i])
        
        # Add new vertices with some initial connections
        for i in range(seed_n, target_n):
            # Connect to a few random existing vertices (check C4 safety)
            candidates = list(range(i))
            rng.shuffle(candidates)
            added = 0
            for v in candidates:
                if added >= 3:
                    break
                if not check_c4_safe(adj, target_n, i, v):
                    adj[i].add(v)
                    adj[v].add(i)
                    added += 1
        
        # Now greedily add edges to boost minimum degree
        stuck_count = 0
        repair_budget = 100
        
        while time.time() - t0 < time_limit:
            md = min(len(adj[i]) for i in range(target_n))
            if md >= min_deg_req:
                # Success! Build edge list
                edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': target_n, 'edges': edges}
            
            # Find vertex with minimum degree
            target = min(range(target_n), key=lambda i: len(adj[i]))
            
            # Try to add an edge from target
            candidates = [v for v in range(target_n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not check_c4_safe(adj, target_n, target, v):
                    adj[target].add(v)
                    adj[v].add(target)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > 5:
                    # Try removing a random edge to unblock
                    if repair_budget > 0:
                        repair_budget -= 1
                        all_edges = [(u, v) for u in range(target_n) for v in adj[u] if u < v]
                        if all_edges:
                            # Prefer edges not incident to target
                            non_target = [(u, v) for u, v in all_edges if u != target and v != target]
                            pool = non_target if non_target else all_edges
                            a, b = rng.choice(pool)
                            adj[a].discard(b)
                            adj[b].discard(a)
                        stuck_count = 0
                    else:
                        break
        
        return None
    
    def greedy_build_fresh(n, min_deg_req, time_limit, rng):
        """Build a C4-free graph from scratch using greedy approach."""
        t0 = time.time()
        adj = [set() for _ in range(n)]
        
        # Initialize with a random spanning tree to ensure connectivity
        order = list(range(n))
        rng.shuffle(order)
        for i in range(1, n):
            u = order[i]
            v = order[rng.randint(0, i - 1)]
            if not check_c4_safe(adj, n, u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        stuck_count = 0
        repair_budget = 200
        
        while time.time() - t0 < time_limit:
            md = min(len(adj[i]) for i in range(n))
            if md >= min_deg_req:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            target = min(range(n), key=lambda i: len(adj[i]))
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not check_c4_safe(adj, n, target, v):
                    adj[target].add(v)
                    adj[v].add(target)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > 5:
                    if repair_budget > 0:
                        repair_budget -= 1
                        all_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                        if all_edges:
                            non_target = [(u, v) for u, v in all_edges if u != target and v != target]
                            pool = non_target if non_target else all_edges
                            a, b = rng.choice(pool)
                            adj[a].discard(b)
                            adj[b].discard(a)
                        stuck_count = 0
                    else:
                        break
        
        return None
    
    def random_repair(n, min_deg_req, time_limit, rng):
        """Generate random graph and repair C4s."""
        t0 = time.time()
        p = (min_deg_req + 3) / (n - 1)
        adj = [set() for _ in range(n)]
        
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Repair: remove C4s
        for _ in range(500):
            if time.time() - t0 > time_limit:
                break
            # Find a C4
            c4_found = False
            for u in range(n):
                if c4_found:
                    break
                for v in range(u + 1, n):
                    if c4_found:
                        break
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        c4_found = True
                        # Remove an edge to break the C4
                        w1, w2 = list(common)[:2]
                        # Remove (u, w1)
                        adj[u].discard(w1)
                        adj[w1].discard(u)
                        break
            
            if not c4_found:
                break
        
        md = min(len(adj[i]) for i in range(n))
        if md >= min_deg_req:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        return None
    
    best_result = None
    best_n = 0
    
    # Try from n=47 down to n=40
    time_budgets = {
        47: 30.0,
        46: 35.0,
        45: 20.0,
        44: 15.0,
        43: 10.0,
        42: 8.0,
        41: 6.0,
        40: 5.0,
        39: 4.0,
    }
    
    seed_counter = 0
    
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        min_deg_req = max(1, n - 39)
        t_limit = min(time_budgets.get(n, 5.0), remaining - 2)
        if t_limit < 1:
            continue
        
        # Strategy 1: Polarity graph seed + greedy extension
        # Find largest q with q^2+q+1 <= n
        q = 1
        while (q + 1) ** 2 + (q + 1) + 1 <= n:
            q += 1
        
        if q >= 2:
            seed_n, seed_adj = build_polarity_graph(q)
            if seed_n <= n:
                # Try extending from polarity graph
                attempts = max(2, int(t_limit / 5.0))
                for attempt in range(attempts):
                    if time.time() - start_time > deadline:
                        break
                    seed = 12345 + seed_counter * 7919 + attempt * 104729
                    seed_counter += 1
                    rng = random.Random(seed)
                    t_alloc = (t_limit / attempts) * 1.5
                    result = greedy_build_from_seed(seed_n, seed_adj, n, min_deg_req, t_alloc, rng)
                    if result is not None:
                        if n > best_n:
                            best_n = n
                            best_result = result
                        return result
        
        # Strategy 2: Fresh greedy build
        attempts = max(2, int(t_limit / 4.0))
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 54321 + seed_counter * 7919 + attempt * 104729
            seed_counter += 1
            rng = random.Random(seed)
            t_alloc = (t_limit / attempts) * 1.2
            result = greedy_build_fresh(n, min_deg_req, t_alloc, rng)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                return result
        
        # Strategy 3: Random repair
        attempts = max(1, int(t_limit / 8.0))
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 98765 + seed_counter * 7919 + attempt * 104729
            seed_counter += 1
            rng = random.Random(seed)
            t_alloc = (t_limit / max(1, attempts))
            result = random_repair(n, min_deg_req, t_alloc, rng)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                return result
    
    # Fallback: return best found or a trivial valid graph
    if best_result is not None:
        return best_result
    
    # Trivial fallback: complete graph on 3 vertices (C4-free, min degree 2)
    return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
