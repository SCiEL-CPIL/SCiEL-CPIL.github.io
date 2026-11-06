import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n = 16
    
    # Grötzsch graph (11 vertices, 20 edges) - 4-chromatic triangle-free
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    core_set = set(core_edges)
    non_core_edges = [e for e in all_pairs if e not in core_set]
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if (adj[u] & adj[v]) != 0:
                        return False
        return True
    
    def is_3_colorable(adj):
        # Backtracking 3-coloring with bitsets
        colors = [-1] * n
        # Order vertices by degree (descending)
        degs = [bin(adj[u]).count('1') for u in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            u = order[idx]
            # Get forbidden colors from neighbors
            forbidden = 0
            for v in range(n):
                if (adj[u] >> v) & 1:
                    if colors[v] >= 0:
                        forbidden |= (1 << colors[v])
            for c in range(3):
                if not (forbidden >> c) & 1:
                    colors[u] = c
                    if backtrack(idx + 1):
                        return True
                    colors[u] = -1
            return False
        
        return backtrack(0)
    
    def edge_count_from_adj(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    def run_greedy(seed, edge_order):
        random.seed(seed)
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        shuffled = edge_order[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def local_search(adj, max_iters=2000):
        # Try edge swaps: remove one edge, add up to two
        best_adj = [a for a in adj]
        best_count = edge_count_from_adj(best_adj)
        
        for it in range(max_iters):
            improved = False
            # Try removing a random edge and adding others
            # Build list of current edges
            cur_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        cur_edges.append((u, v))
            
            if not cur_edges:
                break
            
            # Try 1-for-1 swap
            for _ in range(50):  # limited attempts per iteration
                if time.time() > deadline:
                    return best_adj, best_count
                rem_u, rem_v = random.choice(cur_edges)
                # Try to add a random non-edge
                cand_u, cand_v = random.choice(non_core_edges)
                if cand_u == rem_u or cand_u == rem_v or cand_v == rem_u or cand_v == rem_v:
                    continue
                if (adj[cand_u] >> cand_v) & 1:
                    continue
                # Check if removing (rem_u, rem_v) and adding (cand_u, cand_v) is valid
                # Remove
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                # Check triangle-freeness after removal (should be fine)
                # Try to add
                if (adj[cand_u] & adj[cand_v]) == 0:
                    adj[cand_u] |= (1 << cand_v)
                    adj[cand_v] |= (1 << cand_u)
                    new_count = edge_count_from_adj(adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = [a for a in adj]
                        improved = True
                    # Revert
                    adj[cand_u] &= ~(1 << cand_v)
                    adj[cand_v] &= ~(1 << cand_u)
                # Re-add removed edge
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
            
            if not improved:
                break
        
        return best_adj, best_count
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Random restarts with greedy
    num_restarts = 3000
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        adj = run_greedy(seed, non_core_edges)
        count = edge_count_from_adj(adj)
        if count > best_count:
            # Verify 3-colorability
            if not is_3_colorable(adj):
                best_count = count
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                best_edges = edges
    
    # Phase 2: Local search on best result
    if best_edges:
        # Reconstruct adj from best_edges
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Run local search
        adj, count = local_search(adj, max_iters=5000)
        if count > best_count and not is_3_colorable(adj):
            best_count = count
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            best_edges = edges
    
    # Phase 3: More targeted search - try different orderings
    if time.time() < deadline - 10:
        for seed in range(2000):
            if time.time() > deadline:
                break
            # Use different edge orderings
            random.seed(seed)
            shuffled = non_core_edges[:]
            random.shuffle(shuffled)
            adj = run_greedy(seed, shuffled)
            count = edge_count_from_adj(adj)
            if count > best_count:
                if not is_3_colorable(adj):
                    best_count = count
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    best_edges = edges
    
    return best_edges
