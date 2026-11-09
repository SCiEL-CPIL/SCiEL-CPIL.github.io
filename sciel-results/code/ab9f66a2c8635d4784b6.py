import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_vertices = list(range(n))
    
    # Q3 cube structure
    # Vertices 0-7, edges between vertices differing in exactly one bit
    cube_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                cube_edges.append((i, j))
    
    # cube_edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    
    # Adjacency bitmask
    adj = [0] * n
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count():
        return sum(adj[u].bit_count() for u in range(n)) // 2
    
    # Fast Q3 detection using backtracking with bitmask
    # Check if there exists an injective mapping f: {0..7} -> vertices
    # such that all cube_edges map to graph edges
    def contains_q3():
        # Try all 8-subsets of vertices
        for S in combinations(range(n), 8):
            S_set = set(S)
            S_list = list(S)
            
            # Backtracking: assign cube vertices 0..7 to S_list
            # f[cube_v] = graph_v
            f = [-1] * 8
            used = set()
            
            def backtrack(cube_v):
                if cube_v == 8:
                    # All assigned, verify all cube edges exist
                    for (a, b) in cube_edges:
                        if not has_edge(f[a], f[b]):
                            return False
                    return True
                
                for gv in S_list:
                    if gv in used:
                        continue
                    # Pruning: check edges from already-assigned neighbors
                    ok = True
                    for (a, b) in cube_edges:
                        if a == cube_v and b < cube_v:
                            if not has_edge(gv, f[b]):
                                ok = False
                                break
                        elif b == cube_v and a < cube_v:
                            if not has_edge(gv, f[a]):
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    used.add(gv)
                    f[cube_v] = gv
                    if backtrack(cube_v + 1):
                        return True
                    f[cube_v] = -1
                    used.remove(gv)
                
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    # Faster: check if a specific 8-set S contains Q3
    def q3_in_set(S):
        S_set = set(S)
        S_list = list(S)
        f = [-1] * 8
        used = set()
        
        def backtrack(cube_v):
            if cube_v == 8:
                for (a, b) in cube_edges:
                    if not has_edge(f[a], f[b]):
                        return False
                return True
            
            for gv in S_list:
                if gv in used:
                    continue
                ok = True
                for (a, b) in cube_edges:
                    if a == cube_v and b < cube_v:
                        if not has_edge(gv, f[b]):
                            ok = False
                            break
                    elif b == cube_v and a < cube_v:
                        if not has_edge(gv, f[a]):
                            ok = False
                            break
                if not ok:
                    continue
                
                used.add(gv)
                f[cube_v] = gv
                if backtrack(cube_v + 1):
                    return True
                f[cube_v] = -1
                used.remove(gv)
            
            return False
        
        return backtrack(0)
    
    # Initialize with a known Q3-free graph: K_{4,12} (4*12 = 48 edges)
    # Left: 0,1,2,3. Right: 4..15.
    for u in range(4):
        for v in range(4, 16):
            add_edge(u, v)
    
    current_edges = edge_count()
    best_edges = current_edges
    best_adj = adj[:]
    
    # Phase 1: Greedy edge addition
    # Try to add each missing edge if it doesn't create Q3
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible_edges)
    
    for (u, v) in all_possible_edges:
        if time.time() > deadline - 20:
            break
        if has_edge(u, v):
            continue
        
        add_edge(u, v)
        
        # Check if this creates a Q3
        # Only need to check 8-sets containing both u and v
        others = [x for x in range(n) if x != u and x != v]
        found = False
        for subset6 in combinations(others, 6):
            S = tuple(sorted([u, v] + list(subset6)))
            if q3_in_set(S):
                found = True
                break
        
        if found:
            remove_edge(u, v)
        else:
            current_edges = edge_count()
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
    
    # Phase 2: Local search - try edge swaps (remove one, add another)
    # and single edge additions
    if time.time() < deadline - 10:
        # Rebuild edge list
        current_edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if has_edge(u, v):
                    current_edge_set.add((u, v))
        
        missing_edges = [(u, v) for (u, v) in all_possible_edges if (u, v) not in current_edge_set]
        random.shuffle(missing_edges)
        
        # Try adding missing edges
        for (u, v) in missing_edges:
            if time.time() > deadline - 10:
                break
            if has_edge(u, v):
                continue
            
            add_edge(u, v)
            others = [x for x in range(n) if x != u and x != v]
            found = False
            for subset6 in combinations(others, 6):
                S = tuple(sorted([u, v] + list(subset6)))
                if q3_in_set(S):
                    found = True
                    break
            
            if not found:
                current_edges = edge_count()
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                remove_edge(u, v)
    
    # Phase 3: Simulated annealing-style swaps
    if time.time() < deadline - 5:
        # Try to swap: remove edge e1, add edge e2
        current_edge_list = []
        for u in range(n):
            for v in range(u+1, n):
                if has_edge(u, v):
                    current_edge_list.append((u, v))
        
        missing = [(u, v) for (u, v) in all_possible_edges 
                   if (u, v) not in set(current_edge_list)]
        
        random.shuffle(current_edge_list)
        random.shuffle(missing)
        
        for i in range(min(200, len(current_edge_list))):
            if time.time() > deadline - 5:
                break
            e1 = current_edge_list[i]
            for e2 in missing[:20]:
                if time.time() > deadline - 5:
                    break
                if has_edge(e2[0], e2[1]):
                    continue
                
                # Try the swap
                remove_edge(e1[0], e1[1])
                add_edge(e2[0], e2[1])
                
                # Check Q3 - only need to check sets containing e2
                u2, v2 = e2
                others = [x for x in range(n) if x != u2 and x != v2]
                found = False
                for subset6 in combinations(others, 6):
                    S = tuple(sorted([u2, v2] + list(subset6)))
                    if q3_in_set(S):
                        found = True
                        break
                
                if not found:
                    # Swap was successful, keep it
                    current_edges = edge_count()
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    # Update current_edge_list
                    current_edge_list[i] = e2
                    if e2 in missing:
                        missing.remove(e2)
                    if e1 not in missing:
                        missing.append(e1)
                else:
                    # Revert
                    remove_edge(e2[0], e2[1])
                    add_edge(e1[0], e1[1])
    
    # Restore best solution
    adj[:] = best_adj
    
    # Build edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if has_edge(u, v):
                result.append([u, v])
    
    return result
