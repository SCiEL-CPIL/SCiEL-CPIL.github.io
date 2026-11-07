"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _bfs_diameter(n, adj):
    max_diam = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d == -1:
                return -1
            if d > max_diam:
                max_diam = d
    return max_diam

def _can_add(adj, u, v, n):
    if adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def _is_valid(n, added_edges, base_adj):
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if u == v:
            return False
        if adj[u][v]:
            return False
        if not _can_add(adj, u, v, n):
            return False
        adj[u][v] = True
        adj[v][u] = True
    diam = _bfs_diameter(n, adj)
    return diam <= 4

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    
    # Fallback known valid solution (8 edges, star at 12)
    fallback = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best = list(fallback)
    best_count = len(best)
    
    # In a path, edge (u,v) with |u-v| >= 3 never creates a triangle with base edges.
    # All such edges are valid candidates.
    all_cands = [(u, v) for u in range(n) for v in range(u + 3, n)]
    
    # Try structured "highway" solutions: chains of long edges
    # Pattern: (a0, a1), (a1, a2), ..., (ak-1, ak) where a0=0, ak=23
    # Plus auxiliary edges to cover gaps
    
    # Try 5-edge solutions with specific structures
    
    # Structure 1: 4-edge highway + 1 auxiliary
    # Highway: (0, x1), (x1, x2), (x2, x3), (x3, 23)
    # Need: for all i, dist(i, i+1) <= 4 in augmented graph
    # The highway gives dist(0,23)=4. Other pairs need checking.
    
    # Try specific highway configurations
    highways_4 = [
        [0, 6, 12, 18, 23],
        [0, 5, 10, 15, 23],
        [0, 6, 12, 17, 23],
        [0, 7, 12, 17, 23],
        [0, 5, 11, 17, 23],
        [0, 6, 11, 17, 23],
        [0, 7, 12, 18, 23],
        [0, 5, 10, 16, 23],
        [0, 6, 12, 18, 23],
    ]
    
    for hw in highways_4:
        if time.time() > deadline - 5:
            break
        hw_edges = [(hw[i], hw[i+1]) for i in range(len(hw)-1)]
        # Check if highway edges are valid (no triangles among themselves and with base)
        adj_test = [row[:] for row in base_adj]
        valid_hw = True
        for u, v in hw_edges:
            if not _can_add(adj_test, u, v, n):
                valid_hw = False
                break
            adj_test[u][v] = True
            adj_test[v][u] = True
        
        if not valid_hw:
            continue
        
        # Now try adding 1 auxiliary edge to make diameter <= 4
        # The auxiliary edge should help with the worst-case pairs
        # Try all possible auxiliary edges
        for aux_u, aux_v in all_cands:
            if time.time() > deadline - 2:
                break
            if (aux_u, aux_v) in hw_edges or (aux_v, aux_u) in hw_edges:
                continue
            test_edges = hw_edges + [(aux_u, aux_v)]
            if _is_valid(n, test_edges, base_adj):
                best = test_edges
                best_count = 5
                break
        if best_count == 5:
            break
    
    # Structure 2: 5-edge non-chain solutions
    # Try: 2 hubs with 2 edges each + 1 connecting edge
    if best_count > 5 and time.time() < deadline - 10:
        for h1 in range(4, 20):
            if time.time() > deadline - 5:
                break
            for h2 in range(h1 + 3, n):
                if time.time() > deadline - 5:
                    break
                # Edge (h1, h2) must be valid
                if not _can_add(base_adj, h1, h2, n):
                    continue
                
                # Try: (h1,h2) + 2 edges from h1 + 2 edges from h2
                # Greedy: pick the 2 most useful edges from each hub
                adj_test = [row[:] for row in base_adj]
                adj_test[h1][h2] = True
                adj_test[h2][h1] = True
                
                # Find best 2 edges from h1
                h1_cands = []
                for v in range(n):
                    if v != h1 and v != h2 and not base_adj[h1][v] and _can_add(adj_test, h1, v, n):
                        h1_cands.append(v)
                
                h2_cands = []
                for v in range(n):
                    if v != h2 and v != h1 and not base_adj[h2][v] and _can_add(adj_test, h2, v, n):
                        h2_cands.append(v)
                
                if len(h1_cands) < 2 or len(h2_cands) < 2:
                    continue
                
                # Greedy pick 2 from each
                selected_h1 = []
                temp_adj = [row[:] for row in adj_test]
                for _ in range(2):
                    best_v = None
                    best_score = -1
                    for v in h1_cands:
                        if v in selected_h1:
                            continue
                        if not _can_add(temp_adj, h1, v, n):
                            continue
                        # Score: coverage
                        temp_adj2 = [row[:] for row in temp_adj]
                        temp_adj2[h1][v] = True
                        temp_adj2[v][h1] = True
                        # Count vertices within dist 4 of h1
                        dist = [-1] * n
                        dist[h1] = 0
                        queue = [h1]
                        head = 0
                        count = 0
                        while head < len(queue):
                            u = queue[head]
                            head += 1
                            if dist[u] < 4:
                                count += 1
                                for w in range(n):
                                    if temp_adj2[u][w] and dist[w] == -1:
                                        dist[w] = dist[u] + 1
                                        queue.append(w)
                        if count > best_score:
                            best_score = count
                            best_v = v
                    if best_v is None:
                        break
                    selected_h1.append(best_v)
                    temp_adj[h1][best_v] = True
                    temp_adj[best_v][h1] = True
                
                if len(selected_h1) < 2:
                    continue
                
                selected_h2 = []
                temp_adj2 = [row[:] for row in temp_adj]
                for _ in range(2):
                    best_v = None
                    best_score = -1
                    for v in h2_cands:
                        if v in selected_h2:
                            continue
                        if not _can_add(temp_adj2, h2, v, n):
                            continue
                        temp_adj3 = [row[:] for row in temp_adj2]
                        temp_adj3[h2][v] = True
                        temp_adj3[v][h2] = True
                        dist = [-1] * n
                        dist[h2] = 0
                        queue = [h2]
                        head = 0
                        count = 0
                        while head < len(queue):
                            u = queue[head]
                            head += 1
                            if dist[u] < 4:
                                count += 1
                                for w in range(n):
                                    if temp_adj3[u][w] and dist[w] == -1:
                                        dist[w] = dist[u] + 1
                                        queue.append(w)
                        if count > best_score:
                            best_score = count
                            best_v = v
                    if best_v is None:
                        break
                    selected_h2.append(best_v)
                    temp_adj2[h2][best_v] = True
                    temp_adj2[best_v][h2] = True
                
                if len(selected_h2) < 2:
                    continue
                
                test_edges = [(h1, h2)] + [(h1, v) for v in selected_h1] + [(h2, v) for v in selected_h2]
                if _is_valid(n, test_edges, base_adj):
                    best = test_edges
                    best_count = 5
                    break
            if best_count == 5:
                break
    
    # If no 5-edge solution found, try 6-edge solutions
    if best_count > 5 and time.time() < deadline - 10:
        # Try: 3-edge highway + 3 auxiliary
        for hw in [[0, 8, 16, 23], [0, 7, 15, 23], [0, 8, 15, 23], [0, 6, 14, 23], [0, 8, 16, 23]]:
            if time.time() > deadline - 5:
                break
            hw_edges = [(hw[i], hw[i+1]) for i in range(len(hw)-1)]
            adj_test = [row[:] for row in base_adj]
            valid_hw = True
            for u, v in hw_edges:
                if not _can_add(adj_test, u, v, n):
                    valid_hw = False
                    break
                adj_test[u][v] = True
                adj_test[v][u] = True
            if not valid_hw:
                continue
            
            # Try adding 3 auxiliary edges
            # Use greedy
            aux_edges = []
            temp_adj = [row[:] for row in adj_test]
            for _ in range(3):
                best_aux = None
                best_score = -1
                for u, v in all_cands:
                    if (u, v) in hw_edges or (u, v) in aux_edges:
                        continue
                    if not _can_add(temp_adj, u, v, n):
                        continue
                    temp_adj2 = [row[:] for row in temp_adj]
                    temp_adj2[u][v] = True
                    temp_adj2[v][u] = True
                    d = _bfs_diameter(n, temp_adj2)
                    if d != -1 and d < best_score if best_score > 0 else d != -1:
                        if best_score == -1 or d < best_score:
                            best_score = d
                            best_aux = (u, v)
                    if d == 4:
                        best_aux = (u, v)
                        best_score = 4
                        break
                if best_aux is None:
                    break
                aux_edges.append(best_aux)
                temp_adj[best_aux[0]][best_aux[1]] = True
                temp_adj[best_aux[1]][best_aux[0]] = True
            
            if len(aux_edges) == 3:
                test_edges = hw_edges + aux_edges
                if _is_valid(n, test_edges, base_adj):
                    best = test_edges
                    best_count = 6
                    break
    
    # Final verification
    if not _is_valid(n, best, base_adj):
        best = list(fallback)
    
    return [list(e) for e in best]

# EVOLVE-BLOCK-END
