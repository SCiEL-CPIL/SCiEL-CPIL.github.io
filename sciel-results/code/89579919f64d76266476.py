import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def check_c4_bitmask(adj_bits, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks.
        C4 exists if there's x in N(u)\\{v} and y in N(v)\\{u} with x~y and x!=y.
        Equivalently: N(u) & N(y) has some y in N(v), y != x for x in N(u).
        Simpler: for each x in N(u), check if N(x) & N(v) has any bit other than x itself.
        But even simpler: C4 u-x-y-v means x in N(u), y in N(v), x~y, x!=y.
        This is equivalent to: (N(u) & N(v)) has size >= 2? No, that's a different thing.
        
        Actually: u-x-y-v-u is a C4 iff x in N(u), y in N(v), x~y, x!=y, x!=v, y!=u.
        Since we're adding (u,v), u not in N(v) and v not in N(u) yet.
        So: exists x in N(u), y in N(v) such that x~y and x!=y.
        
        For a given x in N(u): we need y in N(v) with x~y and y!=x.
        So: (N(x) & N(v)) & ~bit(x) != 0 for some x in N(u).
        
        Even better: union over x in N(u) of (N(x) & N(v)) should have a bit that's not in N(u) & N(v) at the same position... 
        Actually simplest: for each x in N(u), if (N(x) & N(v)) has any bit set that isn't x, return True.
        """
        nu = adj_bits[u]
        nv = adj_bits[v]
        # Iterate over set bits in nu
        x = 0
        while x < n:
            # find next set bit
            if nu & (1 << x):
                # Check if N(x) & N(v) has any bit other than x
                nx = adj_bits[x]
                common = nx & nv
                if common & ~(1 << x):
                    return True
            x += 1
        return False

    def check_c4_fast(adj_bits, n, u, v):
        """Faster C4 check: C4 u-x-y-v exists iff 
        there exist x in N(u), y in N(v), x~y, x!=y.
        Equivalent: (OR over x in N(u) of N(x)) & N(v) contains a vertex y such that 
        the x that connected to y is not y itself.
        
        Actually let's just do: for x in N(u), check (N(x) & N(v) & ~bit(x)) != 0
        But we can vectorize: compute W = OR of N(x) for x in N(u), then check W & N(v) & ~(N(u) & N(v))... no that's not right either.
        
        Let me think again. We need x in N(u), y in N(v), x~y, x!=y.
        Let S = {y : exists x in N(u) with x~y and x!=y}
        We need S & N(v) != empty.
        S = (OR_{x in N(u)} N(x)) \\ {x for x in N(u)}... no, S = union of N(x) for x in N(u), minus the x's themselves? No.
        S = {y : y is adjacent to some x in N(u), y != x}
        So S = (union of N(x) for x in N(u)) \\ {x : x in N(u) and x not adjacent to any other x' in N(u)}... this is getting complicated.
        
        Let's just iterate but use bit operations efficiently.
        """
        nu = adj_bits[u]
        nv = adj_bits[v]
        # For each x in N(u), check if (N(x) & nv) has any bit other than x
        x = 0
        while x < n:
            if nu & (1 << x):
                if (adj_bits[x] & nv) & ~(1 << x):
                    return True
            x += 1
        return False

    def polarity_graph(n):
        """Construct a polarity graph which is C4-free.
        For a prime power q, the polarity graph has q^2+1 vertices.
        Vertices are points of PG(2,q) plus the line at infinity.
        Two vertices are adjacent if they are incident in the polarity sense.
        
        For simplicity, we'll use a known construction for specific n values.
        Actually, let's use a simpler approach: start with a random graph and repair.
        
        A simpler C4-free seed: use a bipartite graph with appropriate structure,
        or start from scratch with greedy.
        
        For now, return None to indicate no special seed.
        """
        return None

    def try_build(n, min_deg_req, time_limit, seed):
        rng = random.Random(seed)
        
        # Use bitmasks for adjacency
        adj_bits = np.zeros(n, dtype=np.uint64) if n <= 64 else None
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            if adj_bits is not None:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            if adj_bits is not None:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
        def get_min_deg():
            return min(len(adj[i]) for i in range(n))
            
        def c4_check(u, v):
            """Check if adding (u,v) creates C4"""
            if adj_bits is not None:
                return check_c4_fast(adj_bits, n, u, v)
            else:
                # Fallback for large n (shouldn't happen for n<=47)
                nu = adj[u]
                nv = adj[v]
                for x in nu:
                    for z in adj[x]:
                        if z in nv and z != x:
                            return True
                return False

        # Phase 1: Greedy addition
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        for u in vertices:
            candidates = [v for v in range(n) if v > u and v not in adj[u]]
            rng.shuffle(candidates)
            for v in candidates:
                if not c4_check(u, v):
                    add_edge(u, v)
                    
        # Phase 2: Local Search to maximize min degree
        t0 = time.time()
        stuck_counter = 0
        last_best = get_min_deg()
        best_min_deg = get_min_deg()
        
        while time.time() - t0 < time_limit:
            current_min = get_min_deg()
            if current_min >= min_deg_req:
                return adj
                
            if current_min > best_min_deg:
                best_min_deg = current_min
                stuck_counter = 0
                
            # Find vertex with min degree
            target = min(range(n), key=lambda i: len(adj[i]))
            
            # Try to add an edge to target
            candidates = [v for v in range(n) if v != target and v not in adj[target]]
            rng.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not c4_check(target, v):
                    add_edge(target, v)
                    added = True
                    stuck_counter = 0
                    break
            
            if not added:
                stuck_counter += 1
                if stuck_counter > 30:
                    # Try to remove an edge to unblock
                    # Prefer edges that are "less useful" - edges whose removal 
                    # doesn't reduce min degree too much
                    edges_to_remove = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                # Score: prefer removing edges not incident to low-degree vertices
                                score = 0
                                if len(adj[u]) <= current_min + 1:
                                    score += 10
                                if len(adj[v]) <= current_min + 1:
                                    score += 10
                                edges_to_remove.append((score, u, v))
                    
                    if edges_to_remove:
                        # Remove lowest score edge
                        edges_to_remove.sort(key=lambda x: x[0])
                        _, u, v = edges_to_remove[0]
                        remove_edge(u, v)
                        stuck_counter = 0
                    else:
                        break
                        
        return None

    best_result = None
    
    # Try n from 47 down
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time based on n
        time_budget = 25.0
        if n <= 45:
            time_budget = 15.0
            
        attempts = 8
        success = False
        
        for i in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 1000 + n * 100 + i
            t_limit = time_budget / attempts
            res = try_build(n, min_deg_req, t_limit, seed)
            if res is not None:
                # Verify
                if min(len(res[i]) for i in range(n)) >= min_deg_req:
                    # Check C4 free
                    c4_found = False
                    for u in range(n):
                        for v in res[u]:
                            if u < v:
                                common = res[u].intersection(res[v])
                                if len(common) >= 2:
                                    c4_found = True
                                    break
                        if c4_found: break
                    if not c4_found:
                        edges = []
                        for u in range(n):
                            for v in res[u]:
                                if u < v:
                                    edges.append([u, v])
                        best_result = {'n': n, 'edges': edges}
                        success = True
                        break
        if success:
            return best_result
            
    # Fallback
    if best_result is None:
        return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}
        
    return best_result
