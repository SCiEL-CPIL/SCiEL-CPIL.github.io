"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from functools import lru_cache

_START_TIME = None

def _get_time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound."""
    best = 0
    deadline_time = start_time + deadline
    
    # Precompute adjacency masks
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edge_set.add((u, v))
    
    def is_triangle_free_mask(mask):
        """Check if the set of vertices in mask induces a triangle-free graph."""
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        
        k = len(verts)
        for i in range(k):
            for j in range(i+1, k):
                if (verts[i], verts[j]) not in edge_set:
                    continue
                for l in range(j+1, k):
                    if (verts[i], verts[l]) in edge_set and (verts[j], verts[l]) in edge_set:
                        return False
        return True
    
    # Branch and bound with bitmask
    all_mask = (1 << n) - 1
    
    def dfs(available, current_mask, current_size):
        nonlocal best
        if time.time() > deadline_time:
            return
        if current_size > best:
            best = current_size
        if available == 0:
            return
        
        # Upper bound
        if current_size + bin(available).count('1') <= best:
            return
        
        # Try each vertex in available
        avail = available
        while avail:
            lsb = avail & (-avail)
            v = lsb.bit_length() - 1
            avail ^= lsb
            
            # Check if v can be added to current without forming a triangle
            can_add = True
            cur = current_mask
            while cur:
                lsb2 = cur & (-cur)
                u1 = lsb2.bit_length() - 1
                cur ^= lsb2
                if not (adj_mask[v] & (1 << u1)):
                    continue
                # Check all other vertices in current
                cur2 = current_mask & ~(1 << u1)
                while cur2:
                    lsb3 = cur2 & (-cur2)
                    u2 = lsb3.bit_length() - 1
                    cur2 ^= lsb3
                    if (adj_mask[v] & (1 << u2)) and (adj_mask[u1] & (1 << u2)):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current_mask = current_mask | (1 << v)
                new_available = available & ~(1 << v)
                dfs(new_available, new_current_mask, current_size + 1)
    
    dfs(all_mask, 0, 0)
    return best

def _count_triangles_incremental(adj, n):
    """Count total triangles in the graph for heuristic purposes."""
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if adj[a][c] and adj[b][c]:
                    count += 1
    return count

def _find_largest_triangle_free(adj, n, max_iter=100):
    """
    Heuristic: find a large triangle-free induced subgraph by greedy expansion.
    Returns the size of the largest triangle-free set found.
    """
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    best_size = 0
    best_set = set()
    
    # Try starting from each vertex
    for start_v in range(n):
        current_set = {start_v}
        # Greedily add vertices that don't form triangles
        for v in range(n):
            if v in current_set:
                continue
            # Check if adding v creates a triangle
            creates_triangle = False
            for u in current_set:
                if adj[v][u]:
                    for w in current_set:
                        if w != u and adj[v][w] and adj[u][w]:
                            creates_triangle = True
                            break
                if creates_triangle:
                    break
            if not creates_triangle:
                current_set.add(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
    
    return best_size

def _local_search_bitmask(adj, n, deadline_time, current_best):
    """
    Local search: try edge flips to reduce the max triangle-free induced subgraph.
    Uses incremental evaluation for speed.
    """
    adj = [row[:] for row in adj]
    
    # Precompute adjacency masks
    def get_adj_mask(adj, n):
        masks = [0] * n
        for v in range(n):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            masks[v] = mask
        return masks
    
    adj_mask = get_adj_mask(adj, n)
    
    # Current triangle-free set estimate
    current_tf_size = _find_largest_triangle_free(adj, n)
    
    if current_tf_size >= current_best:
        return adj
    
    # Try edge additions and removals
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline_time:
            break
        
        # Try adding edge (u, v)
        if not adj[u][v]:
            adj[u][v] = True
            adj[v][u] = True
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if _is_k4_free(adj, n):
                new_tf = _find_largest_triangle_free(adj, n)
                if new_tf < current_best:
                    current_best = new_tf
                else:
                    # Revert
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        
        # Try removing edge (u, v)
        if adj[u][v]:
            adj[u][v] = False
            adj[v][u] = False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            
            if _is_k4_free(adj, n):
                new_tf = _find_largest_triangle_free(adj, n)
                if new_tf < current_best:
                    current_best = new_tf
                else:
                    # Revert
                    adj[u][v] = True
                    adj[v][u] = True
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
    
    return adj

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110  # seconds, leave margin for 120s limit
    n = 18
    deadline_time = _START_TIME + deadline
    
    best_edges = None
    best_score = 999
    
    def get_adjacency(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Construction 1: K6,6,6 (complete tripartite) - baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    adj = get_adjacency(edges, n)
    if _is_k4_free(adj, n):
        mtf = _max_triangle_free_induced(adj, n, 3, _START_TIME)
        if mtf < best_score:
            best_score = mtf
            best_edges = edges[:]
    
    # Construction 2: Complete 4-partite with sizes 5,5,4,4
    parts4 = [list(range(0, 5)), list(range(5, 10)), list(range(10, 14)), list(range(14, 18))]
    edges4 = []
    for i in range(4):
        for j in range(i+1, 4):
            for a in parts4[i]:
                for b in parts4[j]:
                    edges4.append([a, b])
    adj4 = get_adjacency(edges4, n)
    if _is_k4_free(adj4, n):
        mtf4 = _max_triangle_free_induced(adj4, n, 3, _START_TIME)
        if mtf4 < best_score:
            best_score = mtf4
            best_edges = edges4[:]
    
    # Construction 3: Turan graph T(18, 5) - complete 5-partite
    parts5 = []
    for i in range(5):
        start = i * (18 // 5) + min(i, 18 % 5)
        end = start + (18 // 5) + (1 if i < 18 % 5 else 0)
        parts5.append(list(range(start, end)))
    edges5 = []
    for i in range(5):
        for j in range(i+1, 5):
            for a in parts5[i]:
                for b in parts5[j]:
                    edges5.append([a, b])
    adj5 = get_adjacency(edges5, n)
    if _is_k4_free(adj5, n):
        mtf5 = _max_triangle_free_induced(adj5, n, 3, _START_TIME)
        if mtf5 < best_score:
            best_score = mtf5
            best_edges = edges5[:]
    
    # Main search: diverse random constructions + local search
    time_budget = 90
    start_search = time.time()
    
    def random_k4_free_graph_v2(n):
        """Generate a random K4-free graph with diverse structure."""
        adj = [[False]*n for _ in range(n)]
        
        # Strategy: start with a random graph, then remove edges to make K4-free
        # Use a biased random graph with varying density
        density = random.uniform(0.3, 0.7)
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Remove edges to ensure K4-free
        for _ in range(200):
            # Find a K4 and remove a random edge from it
            found_k4 = False
            for a in range(n):
                if found_k4:
                    break
                for b in range(a+1, n):
                    if found_k4 or not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if found_k4 or not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4: a,b,c,d
                                edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edges_in_k4)
                                adj[e[0]][e[1]] = False
                                adj[e[1]][e[0]] = False
                                found_k4 = True
                                break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if found_k4:
                    break
            if not found_k4:
                break
        
        return adj
    
    def targeted_local_search(adj, n, deadline_time, current_best):
        """
        Targeted local search: identify the largest triangle-free set and
        try to add edges within it to break it up.
        """
        adj = [row[:] for row in adj]
        
        # Find the largest triangle-free set
        tf_size = _find_largest_triangle_free(adj, n)
        if tf_size < current_best:
            return adj, tf_size
        
        # Find an actual large triangle-free set
        adj_mask = [0] * n
        for v in range(n):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            adj_mask[v] = mask
        
        # Greedily build a large triangle-free set
        best_set = set()
        for start_v in range(n):
            current_set = {start_v}
            for v in range(n):
                if v in current_set:
                    continue
                creates_triangle = False
                for u in current_set:
                    if adj[v][u]:
                        for w in current_set:
                            if w != u and adj[v][w] and adj[u][w]:
                                creates_triangle = True
                                break
                    if creates_triangle:
                        break
                if not creates_triangle:
                    current_set.add(v)
            
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
        
        # Try to add edges within the triangle-free set to break it
        set_list = list(best_set)
        random.shuffle(set_list)
        
        for i in range(len(set_list)):
            for j in range(i+1, len(set_list)):
                if time.time() > deadline_time:
                    break
                u, v = set_list[i], set_list[j]
                if not adj[u][v]:
                    adj[u][v] = True
                    adj[v][u] = True
                    if _is_k4_free(adj, n):
                        new_tf = _find_largest_triangle_free(adj, n)
                        if new_tf < current_best:
                            current_best = new_tf
                            break
                    else:
                        adj[u][v] = False
                        adj[v][u] = False
            if time.time() > deadline_time:
                break
        
        return adj, current_tf_size if tf_size < current_best else _find_largest_triangle_free(adj, n)
    
    for iteration in range(500):
        if time.time() - start_search > time_budget:
            break
        
        # Generate random K4-free graph
        adj_rand = random_k4_free_graph_v2(n)
        
        if not _is_k4_free(adj_rand, n):
            continue
        
        # Quick heuristic evaluation
        tf_heuristic = _find_largest_triangle_free(adj_rand, n)
        
        # If heuristic suggests it's good, do exact evaluation
        if tf_heuristic < best_score + 2:
            mtf_rand = _max_triangle_free_induced(adj_rand, n, 2, _START_TIME)
            if mtf_rand < best_score:
                best_score = mtf_rand
                best_edges = adj_to_edges(adj_rand, n)
                if best_score <= 9:
                    break
        
        # Targeted local search
        if time.time() - start_search < time_budget - 5:
            adj_ls, ls_score = targeted_local_search(adj_rand, n, deadline_time, best_score)
            if _is_k4_free(adj_ls, n):
                mtf_ls = _max_triangle_free_induced(adj_ls, n, 2, _START_TIME)
                if mtf_ls < best_score:
                    best_score = mtf_ls
                    best_edges = adj_to_edges(adj_ls, n)
                    if best_score <= 9:
                        break
    
    # Final: try to improve best_edges with targeted search
    if best_edges is not None and time.time() < deadline_time - 10:
        adj_best = get_adjacency(best_edges, n)
        adj_improved, imp_score = targeted_local_search(adj_best, n, deadline_time, best_score)
        if _is_k4_free(adj_improved, n):
            mtf_imp = _max_triangle_free_induced(adj_improved, n, 3, _START_TIME)
            if mtf_imp < best_score:
                best_score = mtf_imp
                best_edges = adj_to_edges(adj_improved, n)
    
    if best_edges is None:
        best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
