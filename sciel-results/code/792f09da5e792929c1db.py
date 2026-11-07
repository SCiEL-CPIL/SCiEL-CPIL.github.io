import time
import itertools
import random

def construct():
    start_time = time.time()
    timeout = 85.0
    N = 12

    def is_valid_3reg_girth6(edges):
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check 3-regular
        for i in range(N):
            if len(adj[i]) != 3:
                return False, None

        # Check Girth >= 6 (No 3, 4, 5 cycles)
        # No triangles
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    if adj[u].intersection(adj[v]):
                        return False, None
        
        # No 4-cycles
        # A 4-cycle is u-v-w-x-u. u!=w, v!=x.
        # Check all pairs of non-adjacent vertices?
        # Simpler: BFS from each node up to depth 2?
        # If u and v are at distance 2, and they share a neighbor, that's a 3-cycle (already checked).
        # For 4-cycles: u-v-w-x-u.
        # u and w are opposite. dist(u,w)=2.
        # v and x are opposite. dist(v,x)=2.
        # If there are two distinct paths of length 2 between u and w, we have a 4-cycle.
        for u in range(N):
            for w in range(N):
                if u == w or w in adj[u]: continue
                # Find common neighbors
                common = adj[u].intersection(adj[w])
                if len(common) >= 2:
                    return False, None
        
        # No 5-cycles
        # u-v-w-x-y-u
        # u and x are opposite? dist(u,x)=2.
        # v and y are opposite? dist(v,y)=2.
        # w is opposite to...
        # Check: for each u, for each w at dist 2, for each v in N(u), for each x in N(w) \ {v},
        # check if N(v) and N(x) share a neighbor y such that y is not u, w, v, x?
        # This is getting complex. 
        # Since N=12, we can just enumerate all 5-cycles.
        
        # Better: Use the cycle enumeration function to check lengths.
        # But we need a fast filter first.
        # Let's just return True for now and let the cycle checker do the work, 
        # but add a quick 5-cycle check.
        
        # Quick 5-cycle check:
        # For each u, for each v in N(u), for each w in N(v)\{u}, for each x in N(w)\{v},
        # if x in N(u) and x != v: 4-cycle (handled).
        # if N(x) intersects N(u) \ {v, x}:
        #    Let y be in intersection.
        #    Cycle u-v-w-x-y-u.
        #    Check if y != u, v, w, x.
        #    If y is valid, 5-cycle exists.
        
        for u in range(N):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    for x in adj[w]:
                        if x == v or x == u: continue
                        # Check if x is connected to u via a node y
                        # y in N(x) and y in N(u)
                        common_xy_u = adj[x].intersection(adj[u])
                        # y cannot be v (handled by x!=v logic? No, x is neighbor of w, v is neighbor of u)
                        # If y == v, then u-v-x-u is 3-cycle? No, u-v-w-x... 
                        # If y == v, then x is neighbor of v.
                        # Cycle u-v-x-u is 3-cycle. Already checked.
                        # So we look for y != v.
                        for y in common_xy_u:
                            if y == v: continue
                            if y == w: continue # u-v-w-y-u? y=w => u-v-w-w... no
                            if y == x: continue
                            if y == u: continue
                            # If we found such y, we have a 5-cycle u-v-w-x-y-u
                            # Wait, is it simple?
                            # u, v, w, x, y must be distinct.
                            # u != v (edge)
                            # v != w (edge)
                            # w != x (edge)
                            # x != y (edge)
                            # y != u (edge)
                            # u != w? If u==w, 3-cycle u-v-w... no, u-v-u? No.
                            # u != x? If u==x, 4-cycle u-v-w-u... no, u-v-w-x=u. u-v-w-u is 3? No, u-v, v-w, w-u. Triangle. Checked.
                            # v != x? If v==x, 3-cycle u-v-x=u... no. u-v-w-v... no.
                            # v != y? If v==y, 3-cycle u-v-y=x... u-v-x-u. Triangle. Checked.
                            # w != y? If w==y, 3-cycle w-x-y=w... w-x-w? No.
                            # So if distinct, it's a 5-cycle.
                            
                            # We just need to return False if found.
                            return False, None

        return True, adj

    def get_cycle_lengths(adj):
        cycle_lengths = set()
        N = 12
        for start in range(N):
            def dfs(current, visited, length):
                if length >= 3 and current == start:
                    cycle_lengths.add(length)
                    return
                if length > N:
                    return
                for neighbor in adj[current]:
                    if neighbor == start:
                        if length >= 3:
                            cycle_lengths.add(length)
                        continue
                    if neighbor in visited:
                        continue
                    if neighbor < start:
                        continue
                    visited.add(neighbor)
                    dfs(neighbor, visited, length + 1)
                    visited.remove(neighbor)
            
            visited = {start}
            dfs(start, visited, 0)
        return cycle_lengths

    # Generate random 3-regular graphs
    # Use the configuration model or just random matching of half-edges
    # 12 vertices, 3 edges each -> 36 half-edges.
    # Randomly pair them.
    
    best_edges = None
    best_raw = 100
    
    for _ in range(50000):
        if time.time() - start_time > timeout:
            break
            
        # Create list of half edges
        half_edges = []
        for i in range(N):
            for _ in range(3):
                half_edges.append(i)
        
        # Shuffle
        random.shuffle(half_edges)
        
        # Pair them
        edges = []
        has_loop = False
        has_dup = False
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                has_loop = True
                break
            if u > v:
                u, v = v, u
            # Check duplicate
            # Since it's a simple graph check, we can use a set
            if (u, v) in [(a,b) for a,b in edges]:
                has_dup = True
                break
            edges.append((u, v))
        
        if has_loop or has_dup:
            continue
            
        # Check 3-regular and girth >= 6
        valid, adj = is_valid_3reg_girth6(edges)
        if not valid:
            continue
            
        lengths = get_cycle_lengths(adj)
        if not lengths:
            continue
            
        raw = len(lengths)
        if raw < best_raw:
            best_raw = raw
            best_edges = [list(e) for e in edges]
            if best_raw == 1:
                break
                
    if best_edges:
        return best_edges
        
    # Fallback: Two K3,3
    edges = []
    for u in [0,1,2]:
        for v in [3,4,5]:
            edges.append([u,v])
    for u in [6,7,8]:
        for v in [9,10,11]:
            edges.append([u,v])
    return edges
