import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin

    # Fallback: LUW construction
    def luw_fallback():
        q = k
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3:
            d = 3
        
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y) % q
            sub = lambda x,y: (x-y) % q
            mul = lambda x,y: x*y % q
        
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    # Try Voltage Lift of K6
    if k == 5 and g == 9:
        # Base graph K6
        base_n = 6
        base_edges = [(i, j) for i in range(base_n) for j in range(i+1, base_n)]
        # Target size 1116
        # 1116 / 6 = 186
        # Use group Z_186
        group_size = 186
        
        # Random search for voltage assignment
        # Variables: voltage for each edge (i,j) with i<j
        # v[i][j] = v, v[j][i] = -v mod group_size
        
        best_girth = 0
        best_assignment = None
        
        for _ in range(10000): # Try many random assignments
            if time.time() > deadline - 5:
                break
            
            # Random assignment
            voltages = {}
            for i, j in base_edges:
                voltages[(i,j)] = random.randint(0, group_size - 1)
            
            # Build lifted graph
            n_lifted = base_n * group_size
            # We don't need to build full adjacency list if we just check girth
            # But we need to return edges if successful.
            
            # Check girth
            # BFS from each vertex up to depth g//2 - 1 = 3
            # If we find a cycle of length < g, fail.
            
            # To speed up, we can check non-backtracking walks.
            # A cycle of length L < g corresponds to a non-backtracking closed walk of length L.
            # We can just do BFS and check for collisions.
            
            valid = True
            # We need to map (base_node, group_elem) to an ID
            # id = base_node * group_size + group_elem
            
            # For each start vertex, BFS
            for start_base in range(base_n):
                for start_g in range(group_size):
                    # This is too slow: 6 * 186 = 1116 starts.
                    # BFS depth 3: 1 + 5 + 20 + 80 = 106 nodes.
                    # 1116 * 106 ~ 118k operations.
                    # 10000 trials * 118k = 1.18e9. Too slow in Python.
                    
                    pass
            
            # Optimization: We only need to check if *any* cycle < 9 exists.
            # We can pick a few random start vertices.
            # Or better: use the property that if there is a short cycle, it's likely to be found quickly.
            
            # Let's try a smarter check:
            # Generate all non-backtracking walks of length 4 from a random set of vertices.
            # If any walk closes, we have a cycle <= 8 (actually <= 4+1? No, closed walk length L).
            # Girth 9 means no closed non-backtracking walk of length < 9.
            
            # Let's just try 100 random trials with a fast check.
            # Fast check:
            # Pick 50 random start vertices.
            # Expand non-backtracking walks of length 4.
            # If any (u, v) is visited twice (with different history?), we might have a cycle.
            # Actually, just checking for any repeated vertex in the BFS tree of depth 3 is enough to detect cycles <= 2*3+1 = 7?
            # No, cycle length L implies distance L/2.
            # If girth >= 9, radius 3 is a tree.
            # So if BFS depth 3 from any vertex has a collision (same node reached by 2 paths), girth < 9.
            
            # We can just check a subset of vertices.
            
            trial_valid = True
            # Sample 20 random start vertices
            starts = random.sample(range(n_lifted), 20)
            for s in starts:
                s_base = s // group_size
                s_g = s % group_size
                
                visited = {s: 0} # node: parent
                # BFS
                queue = deque([(s, -1, -1)]) # node, prev_node, prev_edge_id
                # Actually, we need to track the path to ensure non-backtracking?
                # For girth check, simple BFS is fine. If we find a neighbor that is already visited
                # and is not the parent, we have a cycle.
                # Length of cycle = depth[u] + depth[v] + 1.
                # We want min cycle length >= 9.
                
                depths = {s: 0}
                parents = {s: -1}
                q = deque([s])
                
                while q and trial_valid:
                    u = q.popleft()
                    u_base = u // group_size
                    u_g = u % group_size
                    
                    for v_base, v_g in get_neighbours_lifted(u_base, u_g, voltages, group_size):
                        v = v_base * group_size + v_g
                        if v == parents.get(u, -1):
                            continue
                        if v in depths:
                            # Found a cycle
                            # Cycle length = depths[u] + depths[v] + 1
                            # But this might not be the shortest.
                            # However, if depths[u] + depths[v] + 1 < 9, we fail.
                            # If depths[u] + depths[v] + 1 >= 9, it might be ok, but we should check.
                            # Actually, if we find *any* cycle < 9, we fail.
                            # The first collision in BFS gives the shortest cycle through that edge.
                            cycle_len = depths[u] + depths[v] + 1
                            if cycle_len < g:
                                trial_valid = False
                                break
                        else:
                            depths[v] = depths[u] + 1
                            parents[v] = u
                            if depths[u] < 4: # Only expand up to depth 3 or 4?
                                # If girth is 9, we need to check up to depth 4?
                                # Radius 3 is tree. Depth 4 might show cycles of length 9.
                                # We want to ensure NO cycle < 9.
                                # So if we find a cycle of length 5, 6, 7, 8, we fail.
                                # BFS finds shortest cycles.
                                # So if we find a collision at depth d1, d2, cycle len = d1+d2+1.
                                # If this is < 9, fail.
                                q.append(v)
                
                if not trial_valid:
                    break
            
            if trial_valid:
                # We think it's valid.
                # Let's do a full verification to be sure.
                full_valid = True
                for s in range(n_lifted):
                    # Full BFS
                    depths = {s: 0}
                    parents = {s: -1}
                    q = deque([s])
                    while q and full_valid:
                        u = q.popleft()
                        u_base = u // group_size
                        u_g = u % group_size
                        for v_base, v_g in get_neighbours_lifted(u_base, u_g, voltages, group_size):
                            v = v_base * group_size + v_g
                            if v == parents.get(u, -1):
                                continue
                            if v in depths:
                                cycle_len = depths[u] + depths[v] + 1
                                if cycle_len < g:
                                    full_valid = False
                                    break
                            else:
                                depths[v] = depths[u] + 1
                                parents[v] = u
                                if depths[u] < 5:
                                    q.append(v)
                    if not full_valid:
                        break
                
                if full_valid:
                    # Construct edges
                    edges = []
                    for i in range(base_n):
                        for j in range(i+1, base_n):
                            v_ij = voltages[(i,j)]
                            # Edge between (i, g) and (j, g+v_ij)
                            # And (j, g) and (i, g-v_ij)
                            # We need to list all edges in the lifted graph
                            # To avoid duplicates, we can iterate over all i, j, g
                            # and add edge if i < j or (i==j and g1 < g2)
                            # But i != j in K6.
                            for g_elem in range(group_size):
                                u = i * group_size + g_elem
                                v = j * group_size + (g_elem + v_ij) % group_size
                                # Ensure u < v for consistency? No, just collect unique edges.
                                # We can use a set of frozensets or tuples (min, max)
                                if u < v:
                                    edges.append((u, v))
                                else:
                                    edges.append((v, u))
                            # The other direction is covered when we swap i,j?
                            # No, we only iterate i<j.
                            # The edge (j, g) to (i, g-v_ij) is the same as (i, g-v_ij) to (j, g).
                            # Let g' = g - v_ij.
                            # Then (i, g') to (j, g'+v_ij) = (j, g).
                            # So iterating g from 0 to group_size-1 covers all edges.
                    
                    # Remove duplicates
                    edges = list(set(edges))
                    # Verify degree
                    # Return
                    return n_lifted, edges

    # If no valid lift found, return LUW
    return luw_fallback()

def get_neighbours_lifted(u_base, u_g, voltages, group_size):
    # K6 neighbours are all other nodes
    for v_base in range(6):
        if v_base == u_base:
            continue
        # Get voltage
        if u_base < v_base:
            v_val = voltages[(u_base, v_base)]
        else:
            v_val = -voltages[(v_base, u_base)]
        
        v_g = (u_g + v_val) % group_size
        yield v_base, v_g
