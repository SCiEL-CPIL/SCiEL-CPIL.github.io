import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def _precompute_path4(v, adj_list):
        """
        Precompute Path4[c][i][d] for all c, i, d < v.
        Path4[c][i][d] is True if there is a simple path of length 4 
        between i and d in the graph G_c induced by vertices 0..v-1.
        """
        Path4 = [ [[False]*v for _ in range(v)] for _ in range(4) ]
        for c in range(4):
            for i in range(v):
                if not adj_list[c][i]:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    # Path i-a-b-e-d
                    # a is neighbor of i
                    for a in adj_list[c][i]:
                        if a == d:
                            continue
                        # b is neighbor of a
                        for b in adj_list[c][a]:
                            if b == i or b == d or b == a:
                                continue
                            # e is neighbor of b
                            for e in adj_list[c][b]:
                                if e == i or e == d or e == a or e == b:
                                    continue
                                # Check if e is connected to d
                                if d in adj_list[c][e]:
                                    found = True
                                    break
                        if found:
                            break
                    Path4[c][i][d] = found
        return Path4

    def _constrained_backtrack(n_target, deadline):
        adj = [[-1]*n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            # Precompute Path4 for the current subgraph 0..v-1
            # This is done once per level v
            Path4 = _precompute_path4(v, adj_list)
            
            def _assign(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    return _search(v + 1)
                
                i = idx
                for c in range(4):
                    # Check if setting adj[i][v]=c creates a C6
                    # A C6 would involve v, i, and 4 other vertices in a path i-a-b-e-d
                    # where d is another neighbor of v in color c.
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c:
                            if Path4[c][i][d]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    
                    if _assign(idx + 1):
                        return True
                    
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                
                return False
            
            return _assign(0)
        
        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj[i][j])
            return colors
        return None

    def _verify(colors, n):
        if n < 6:
            return True
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Re-verify using the same Path4 logic or a simple check
        # For safety, use a brute force check on the final result
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        for c in range(4):
            # Check for C6 in color c
            # We can reuse _precompute_path4 logic or just check all pairs
            # If there is a C6, there exists a pair (i, d) with a path of length 4
            # But wait, the C6 is a cycle, so i and d are connected by the path AND by the cycle closure?
            # No, the cycle is i-a-b-e-d-...-i? No.
            # The cycle is v-u1-u2-u3-u4-u5-v.
            # In the final graph, we just need to check if any color class contains a C6.
            # My Path4 check was for "path of length 4 between two neighbors of v".
            # This forms a C6 when v is added.
            # In the final graph, a C6 is any 6-cycle.
            # We can check if there exists any i, d such that Path4[c][i][d] is true 
            # AND i, d are NOT connected by an edge? No.
            # If Path4[c][i][d] is true, there is a path i-a-b-e-d.
            # If i and d are connected by an edge, it's a C5 or C4 etc.
            # If i and d are NOT connected, it's just a path.
            # To form a C6, we need a path of length 5 between two adjacent vertices?
            # Or a path of length 4 between two vertices that are connected?
            # No, a C6 is a cycle of length 6.
            # It consists of 6 edges.
            # If we remove one edge (u, v), we have a path of length 5 between u and v.
            # So, a C6 exists iff there exist adjacent vertices u, v in G_c 
            # such that there is a simple path of length 5 between u and v in G_c - (u, v).
            
            # Let's just do a direct check for C6 in the final graph to be safe.
            # n is at most 20. C(20, 6) = 38760.
            # For each subset, check if it contains a C6.
            # This is fast enough for verification.
            
            for subset in combinations(range(n), 6):
                # Check if the induced subgraph on subset contains a C6
                # A 6-vertex graph contains a C6 iff it has a Hamiltonian cycle.
                # There are 60 Hamiltonian cycles on 6 labeled vertices.
                # We can check if any of them is present.
                
                # Precompute the 60 cycles on 0..5
                if not hasattr(_verify, '_cycles'):
                    cycles = []
                    seen = set()
                    for perm in permutations(range(1, 6)):
                        cycle = (0,) + perm
                        edges = frozenset(frozenset((cycle[k], cycle[(k+1)%6])) for k in range(6))
                        if edges not in seen:
                            seen.add(edges)
                            cycles.append(cycle)
                    _verify._cycles = cycles
                
                for cycle in _verify._cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k+1)%6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    # Target n values
    for target_n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this target_n
        found = None
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            # Reset random seed? No, we want diversity.
            res = _constrained_backtrack(target_n, deadline)
            if res:
                # Extract colors
                if _verify(res, target_n):
                    return {'n': target_n, 'colors': res}
                else:
                    # Verification failed, continue searching
                    continue
            if attempts > 5: # Limit attempts per n to avoid hanging on hard instances
                break
            
    # Fallback to known good solution (n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
