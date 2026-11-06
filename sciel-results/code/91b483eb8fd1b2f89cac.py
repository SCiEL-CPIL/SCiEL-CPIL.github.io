"""Improved program for K4-free graph construction with fast evaluation."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

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

def _max_triangle_free_induced_fast(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound with bitmasks."""
    deadline_time = start_time + deadline
    
    # Precompute adjacency as bitmasks
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
    
    # Precompute for each vertex, the set of vertices that form a triangle with it
    # tri_conflict[v] = set of vertices u such that there exists w with adj[v][w] and adj[u][w] and adj[v][u]
    # Actually, we need: v and u are "in conflict" if there's a w adjacent to both v and u
    # But for triangle-free, we need: no triple (a,b,c) all mutually adjacent
    
    # Better approach: for a set S to be triangle-free, for every pair (u,v) in S that are adjacent,
    # there must be no w in S adjacent to both u and v.
    
    # Precompute common_neighbors[u][v] = bitmask of vertices adjacent to both u and v
    common = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                common[u][v] = common[v][u] = adj_mask[u] & adj_mask[v]
    
    best = [0]
    
    def dfs(available_mask, current_mask, current_size):
        if time.time() > deadline_time:
            return
        if current_size > best[0]:
            best[0] = current_size
        if available_mask == 0:
            return
        
        # Upper bound
        if current_size + bin(available_mask).count('1') <= best[0]:
            return
        
        # Try adding vertices from available
        avail = available_mask
        while avail:
            v_bit = avail & (-avail)
            avail ^= v_bit
            v = v_bit.bit_length() - 1
            
            # Check if v can be added: for all u in current, if adj[v][u], then no w in current with adj[v][w] and adj[u][w]
            can_add = True
            cur = current_mask
            while cur:
                u_bit = cur & (-cur)
                cur ^= u_bit
                u = u_bit.bit_length() - 1
                if adj[v][u]:
                    # Check if any w in current is adjacent to both v and u
                    common_vu = common[v][u] & current_mask
                    if common_vu:
                        can_add = False
                        break
            
            if can_add:
                new_current = current_mask | v_bit
                new_available = available_mask & ~v_bit
                dfs(new_available, new_current, current_size + 1)
    
    dfs((1 << n) - 1, 0, 0)
    return best[0]

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 140  # seconds, leave margin for 150s limit
    n = 18
    
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
    
    def is_k4_free_fast(adj, n):
        """Fast K4-free check."""
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
    
    # Start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    adj = get_adjacency(edges, n)
    if _is_k4_free(adj, n):
        mtf = _max_triangle_free_induced_fast(adj, n, 2, _START_TIME)
        if mtf < best_score:
            best_score = mtf
            best_edges = edges[:]
    
    # Local search: try to improve by adding edges within parts
    # Key insight: adding edges within a part can break triangle-free induced subgraphs
    # that span two parts
    
    def try_local_search(adj, current_score, time_budget):
        """Perform local search to improve the score."""
        nonlocal best_score, best_edges
        n = 18
        start_time = time.time()
        current_adj = [row[:] for row in adj]
        current_score = current_score
        
        # Generate candidate edges to add (within parts)
        candidate_edges = []
        for part in parts:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    u, v = part[i], part[j]
                    if not current_adj[u][v]:
                        candidate_edges.append((u, v))
        
        # Also consider removing edges between parts to break triangles
        inter_part_edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        if current_adj[a][b]:
                            inter_part_edges.append((a, b))
        
        random.shuffle(candidate_edges)
        random.shuffle(inter_part_edges)
        
        # Try adding candidate edges one by one
        for u, v in candidate_edges:
            if time.time() - start_time > time_budget:
                break
            if time.time() > _START_TIME + deadline:
                break
            
            # Check if adding edge (u,v) keeps graph K4-free
            current_adj[u][v] = True
            current_adj[v][u] = True
            
            if not is_k4_free_fast(current_adj, n):
                current_adj[u][v] = False
                current_adj[v][u] = False
                continue
            
            new_score = _max_triangle_free_induced_fast(current_adj, n, 1.5, _START_TIME)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = adj_to_edges(current_adj, n)
            else:
                # Revert
                current_adj[u][v] = False
                current_adj[v][u] = False
        
        return current_score, current_adj
    
    # Try multiple random initial perturbations
    num_tries = 20
    for trial in range(num_tries):
        if time.time() > _START_TIME + deadline - 5:
            break
        
        # Create a perturbed version of K6,6,6
        adj_try = [row[:] for row in adj]
        
        # Randomly add some within-part edges
        num_add = random.randint(3, 8)
        added = set()
        for _ in range(num_add * 3):
            part_idx = random.randint(0, 2)
            part = parts[part_idx]
            i, j = random.sample(range(len(part)), 2)
            u, v = part[i], part[j]
            if (u, v) not in added and (v, u) not in added:
                adj_try[u][v] = True
                adj_try[v][u] = True
                added.add((u, v))
                added.add((v, u))
        
        # Randomly remove some between-part edges
        num_remove = random.randint(0, 5)
        removed = set()
        for _ in range(num_remove * 3):
            i, j = random.sample(range(3), 2)
            a = random.choice(parts[i])
            b = random.choice(parts[j])
            if (a, b) not in removed and (b, a) not in removed:
                if adj_try[a][b]:
                    adj_try[a][b] = False
                    adj_try[b][a] = False
                    removed.add((a, b))
                    removed.add((b, a))
        
        if not is_k4_free_fast(adj_try, n):
            continue
        
        score_try = _max_triangle_free_induced_fast(adj_try, n, 2, _START_TIME)
        if score_try < best_score:
            best_score = score_try
            best_edges = adj_to_edges(adj_try, n)
        
        # Local search from this point
        if time.time() < _START_TIME + deadline - 10:
            score_try, adj_try = try_local_search(adj_try, score_try, 3)
    
    # Additional random search
    if time.time() < _START_TIME + deadline - 10:
        random.seed(42)
        for iteration in range(5000):
            if time.time() > _START_TIME + deadline - 5:
                break
            
            # Generate random K4-free graph
            adj_rand = [[False]*n for _ in range(n)]
            perm = list(range(n))
            random.shuffle(perm)
            sizes = [6, 6, 6]
            rand_parts = []
            idx = 0
            for s in sizes:
                rand_parts.append(perm[idx:idx+s])
                idx += s
            
            for i in range(3):
                for j in range(i+1, 3):
                    for a in rand_parts[i]:
                        for b in rand_parts[j]:
                            if random.random() < 0.85:
                                adj_rand[a][b] = True
                                adj_rand[b][a] = True
            
            for part in rand_parts:
                for i in range(len(part)):
                    for j in range(i+1, len(part)):
                        if random.random() < 0.25:
                            adj_rand[part[i]][part[j]] = True
                            adj_rand[part[j]][part[i]] = True
            
            # Fix K4 violations
            max_fixes = 20
            for _ in range(max_fixes):
                if is_k4_free_fast(adj_rand, n):
                    break
                found = False
                for a in range(n):
                    for b in range(a+1, n):
                        if not adj_rand[a][b]:
                            continue
                        for c in range(b+1, n):
                            if not adj_rand[a][c] or not adj_rand[b][c]:
                                continue
                            for d in range(c+1, n):
                                if adj_rand[a][d] and adj_rand[b][d] and adj_rand[c][d]:
                                    edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                    e = random.choice(edges_in_k4)
                                    adj_rand[e[0]][e[1]] = False
                                    adj_rand[e[1]][e[0]] = False
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    break
            
            if not is_k4_free_fast(adj_rand, n):
                continue
            
            score_rand = _max_triangle_free_induced_fast(adj_rand, n, 1.5, _START_TIME)
            if score_rand < best_score:
                best_score = score_rand
                best_edges = adj_to_edges(adj_rand, n)
                if best_score <= 9:
                    break
    
    if best_edges is None:
        best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
