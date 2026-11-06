import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def check_c4_safe(adj, u, v):
        """
        Check if adding edge (u,v) creates a C4.
        Returns True if safe, False if C4 is created.
        """
        nu = adj[u]
        nv = adj[v]
        # We look for x in nv, y in nu such that x != y and (x,y) is an edge.
        # Cycle: u-v-x-y-u
        if len(nu) > len(nv):
            nu, nv = nv, nu
            
        for y in nu:
            for x in nv:
                if y == x:
                    continue
                if x in adj[y]:
                    return False
        return True

    def find_conflicting_c4(adj, u, v):
        """
        Find a specific C4 that would be created by adding (u,v).
        Returns a list of 3 existing edges forming the path u-y-x-v or similar.
        """
        nu = adj[u]
        nv = adj[v]
        if len(nu) > len(nv):
            nu, nv = nv, nu
            
        for y in nu:
            for x in nv:
                if y == x:
                    continue
                if x in adj[y]:
                    # C4 is u-v-x-y-u
                    # Existing edges: (v,x), (x,y), (y,u)
                    return [(v, x), (x, y), (y, u)]
        return None

    def greedy_build(n, min_deg_req, time_limit, seed):
        t0 = time.time()
        rng = random.Random(seed)
        
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            
        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            
        def min_degree():
            return min(len(adj[i]) for i in range(n))
            
        def get_edges_list():
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return edges

        stuck_count = 0
        max_stuck = 50
        
        while time.time() - t0 < time_limit:
            md = min_degree()
            if md >= min_deg_req:
                return {'n': n, 'edges': get_edges_list()}
                
            # Find vertex with min degree
            target = min(range(n), key=lambda i: len(adj[i]))
            
            # Candidates for connection
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            if not candidates:
                # Target is connected to everyone.
                # If min_deg < req, we are stuck.
                # But as analyzed, this shouldn't happen if n is large enough and we add edges.
                # If it happens, we break.
                break

            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if check_c4_safe(adj, target, v):
                    add_edge(target, v)
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                if stuck_count > max_stuck:
                    # Repair: Find a conflicting C4 for one of the candidates and remove an edge from it
                    v = rng.choice(candidates)
                    c4_edges = find_conflicting_c4(adj, target, v)
                    if c4_edges:
                        # Remove one of the edges in the C4
                        edge_to_remove = rng.choice(c4_edges)
                        remove_edge(edge_to_remove[0], edge_to_remove[1])
                    else:
                        # Fallback: remove random edge
                        all_edges = []
                        for u in range(n):
                            for w in adj[u]:
                                if u < w:
                                    all_edges.append((u,w))
                        if all_edges:
                            e = rng.choice(all_edges)
                            remove_edge(e[0], e[1])
                    
                    stuck_count = 0

        return None

    best_result = None
    best_n = 0
    
    time_budgets = {
        47: 30.0,
        46: 40.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 5.0,
    }
    
    seed_counter = 0
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        t_limit = min(time_budgets.get(n, 5.0), remaining - 1)
        if t_limit < 1:
            continue
            
        attempts = max(1, int(t_limit / 0.5))
        
        for attempt in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 1000 + seed_counter * 7919 + attempt
            seed_counter += 1
            # Allocate remaining time for this attempt
            # If it's the last attempt, give it the rest of the budget
            if attempt == attempts - 1:
                att_limit = t_limit - (time.time() - (start_time + (deadline - t_limit))) # This is getting complicated
                # Simpler: just use a fixed small slice or the remaining time in the budget for n
                # Let's just use t_limit / (attempts - attempt)
                att_limit = t_limit / 1.0
            else:
                att_limit = t_limit / (attempts - attempt)
                
            # Ensure att_limit is positive
            if att_limit <= 0:
                break
                
            result = greedy_build(n, min_deg_req, att_limit, seed)
            if result is not None:
                return result
                
    return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}
