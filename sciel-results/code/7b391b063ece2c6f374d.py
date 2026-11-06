import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_edges = len(all_edges)
    
    # Precompute for faster Q3 checks
    # For each pair (u,v), precompute the list of "Q3 patterns" that include (u,v)
    # A Q3 containing edge (u,v): the other 6 vertices must satisfy specific adjacency constraints
    
    def has_q3_with_edge(adj, u, v):
        # Check if current adj (with u-v already present) contains a Q3 using edge (u,v)
        # Q3 structure with u~v:
        #   u has 2 other Q3-neighbors: a, b
        #   v has 2 other Q3-neighbors: c, d  
        #   a~c XOR a~d, b~c XOR b~d (exactly one each)
        #   There exist e, f such that:
        #     e~a, e~b, e~(c or d), f~a, f~b, f~(d or c), e~f
        #     e and f are the remaining 2 vertices
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        if len(nu_list) < 2 or len(nv_list) < 2:
            return False
        
        # Precompute adj bits for speed
        adj_list = adj  # already a list of ints
        
        for i in range(len(nu_list)):
            a = nu_list[i]
            for j in range(i+1, len(nu_list)):
                b = nu_list[j]
                ab_mask = adj_list[a] & adj_list[b]
                for k in range(len(nv_list)):
                    c = nv_list[k]
                    for l in range(k+1, len(nv_list)):
                        d = nv_list[l]
                        if a == c or a == d or b == c or b == d:
                            continue
                        ac = (adj_list[a] >> c) & 1
                        ad = (adj_list[a] >> d) & 1
                        if ac + ad != 1:
                            continue
                        bc = (adj_list[b] >> c) & 1
                        bd = (adj_list[b] >> d) & 1
                        if bc + bd != 1:
                            continue
                        # Find e, f in N(a)∩N(b) \ {u,v,c,d}
                        ab_common = ab_mask & ~(1 << u) & ~(1 << v) & ~(1 << c) & ~(1 << d)
                        if ab_common == 0:
                            continue
                        # Get candidate list
                        candidates = [x for x in range(n) if (ab_common >> x) & 1]
                        for ei in range(len(candidates)):
                            e = candidates[ei]
                            ec = (adj_list[e] >> c) & 1
                            ed = (adj_list[e] >> d) & 1
                            if ec + ed != 1:
                                continue
                            for fi in range(ei+1, len(candidates)):
                                f = candidates[fi]
                                # f must connect to the other of c,d
                                if ec == 1:
                                    if not ((adj_list[f] >> d) & 1):
                                        continue
                                else:
                                    if not ((adj_list[f] >> c) & 1):
                                        continue
                                if not ((adj_list[e] >> f) & 1):
                                    continue
                                return True
        return False

    def run_greedy(seed, init_density=1.0):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            if init_density < 1.0 and random.random() > init_density:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search: try to add missing edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            try_order = all_edges[:]
            random.shuffle(try_order)
            for u, v in try_order:
                if (u, v) in edge_set:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return count, list(edge_set)

    def run_greedy_with_swaps(seed):
        """Greedy + local search with edge swaps to escape local optima"""
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search with swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # First try adding edges
            try_order = all_edges[:]
            random.shuffle(try_order)
            for u, v in try_order:
                if (u, v) in edge_set:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            # Try swaps: remove one edge, add another
            if time.time() < deadline:
                current_edges = list(edge_set)
                random.shuffle(current_edges)
                for rem_u, rem_v in current_edges[:20]:  # Limit swaps tried
                    if time.time() >= deadline:
                        break
                    # Remove
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    edge_set.discard((rem_u, rem_v))
                    count -= 1
                    
                    # Try to add a different edge
                    added = False
                    for add_u, add_v in all_edges:
                        if (add_u, add_v) in edge_set or (add_v, add_u) in edge_set:
                            continue
                        adj[add_u] |= (1 << add_v)
                        adj[add_v] |= (1 << add_u)
                        if not has_q3_with_edge(adj, add_u, add_v):
                            edge_set.add((add_u, add_v))
                            count += 1
                            added = True
                            break
                        else:
                            adj[add_u] &= ~(1 << add_v)
                            adj[add_v] &= ~(1 << add_u)
                    
                    if not added:
                        # Restore
                        adj[rem_u] |= (1 << rem_v)
                        adj[rem_v] |= (1 << rem_u)
                        edge_set.add((rem_u, rem_v))
                        count += 1
                    else:
                        improved = True
        
        return count, list(edge_set)

    best_count = 0
    best_edges = []
    
    restart = 0
    while time.time() < deadline:
        if restart % 3 == 0:
            count, edges_list = run_greedy(restart)
        elif restart % 3 == 1:
            count, edges_list = run_greedy_with_swaps(restart)
        else:
            count, edges_list = run_greedy(restart, init_density=0.7)
        
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
    
    return [list(e) for e in best_edges]
