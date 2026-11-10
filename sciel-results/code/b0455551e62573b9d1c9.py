import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55
    
    def is_valid(adj, deg):
        for s in range(n):
            for u in adj[s]:
                for v in adj[u]:
                    if v == s:
                        continue
                    for t in adj[v]:
                        if t == s or t == u:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def try_add_edge(adj, deg, u, v):
        # Case 1: s-u-v-t
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # Case 3: u-v-x-y
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[u] == deg[y]:
                    return False
        # Case 4: v-u-x-y
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[v] == deg[y]:
                    return False
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        return True
    
    def greedy_fill(adj, deg, edge_order):
        for u, v in edge_order:
            if u in adj[v]:
                continue
            try_add_edge(adj, deg, u, v)
    
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2
    
    def local_search_swap(adj, deg, all_edges, max_iter=300):
        best_edges = count_edges(adj)
        best_adj = [set(s) for s in adj]
        best_deg = deg[:]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            current_edge_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            missing_edge_list = [(u, v) for u, v in all_edges if u not in adj[v]]
            
            if not current_edge_list or not missing_edge_list:
                break
            
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                rem_u, rem_v = random.choice(current_edge_list)
                add_u, add_v = random.choice(missing_edge_list)
                
                save_adj = [set(s) for s in adj]
                save_deg = deg[:]
                
                adj[rem_u].discard(rem_v)
                adj[rem_v].discard(rem_u)
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                if try_add_edge(adj, deg, add_u, add_v):
                    new_edges = count_edges(adj)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [set(s) for s in adj]
                        best_deg = deg[:]
                        current_edge_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                        missing_edge_list = [(u, v) for u, v in all_edges if u not in adj[v]]
                    else:
                        for i in range(n):
                            adj[i] = set(save_adj[i])
                        deg = save_deg[:]
                else:
                    for i in range(n):
                        adj[i] = set(save_adj[i])
                    deg = save_deg[:]
        
        return best_adj, best_deg, best_edges
    
    best_adj = None
    best_deg = None
    best_edges = 0
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Baseline: K_{7,8}
    adj = [set() for _ in range(n)]
    deg = [0] * n
    part_a = list(range(7))
    part_b = list(range(7, 15))
    for u in part_a:
        for v in part_b:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
    best_edges = count_edges(adj)
    best_adj = [set(s) for s in adj]
    best_deg = deg[:]
    
    # Try various bipartite splits
    for a_size in range(4, 12):
        if time.time() > deadline:
            break
        if a_size == n - a_size:
            continue
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        part_a = list(range(a_size))
        part_b = list(range(a_size, n))
        for u in part_a:
            for v in part_b:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        remaining = [(u, v) for u, v in all_possible_edges if u not in adj[v]]
        random.shuffle(remaining)
        greedy_fill(adj, deg, remaining)
        
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
    
    # Randomized greedy with diverse orderings
    for attempt in range(800):
        if time.time() > deadline - 10:
            break
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        random.seed(attempt * 31 + 7)
        
        # Random initial seed
        initial_count = random.randint(15, 55)
        initial_edges = random.sample(all_possible_edges, initial_count)
        for u, v in initial_edges:
            if try_add_edge(adj, deg, u, v):
                pass
        
        remaining = [(u, v) for u, v in all_possible_edges if u not in adj[v]]
        random.shuffle(remaining)
        greedy_fill(adj, deg, remaining)
        
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
    
    # Local search from best
    if time.time() < deadline - 5:
        curr_adj = [set(s) for s in best_adj]
        curr_deg = best_deg[:]
        curr_adj, curr_deg, curr_edges = local_search_swap(curr_adj, curr_deg, all_possible_edges, max_iter=400)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = curr_adj
            best_deg = curr_deg
    
    # Perturbation + local search
    for restart in range(30):
        if time.time() > deadline - 2:
            break
        
        curr_adj = [set(s) for s in best_adj]
        curr_deg = best_deg[:]
        
        current_edge_list = [(u, v) for u in range(n) for v in curr_adj[u] if u < v]
        k = random.randint(3, 10)
        if len(current_edge_list) >= k:
            to_remove = random.sample(current_edge_list, k)
            for u, v in to_remove:
                curr_adj[u].discard(v)
                curr_adj[v].discard(u)
                curr_deg[u] -= 1
                curr_deg[v] -= 1
        
        remaining = [(u, v) for u, v in all_possible_edges if u not in curr_adj[v]]
        random.shuffle(remaining)
        greedy_fill(curr_adj, curr_deg, remaining)
        
        curr_adj, curr_deg, curr_edges = local_search_swap(curr_adj, curr_deg, all_possible_edges, max_iter=150)
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = curr_adj
            best_deg = curr_deg
    
    # Verify
    if best_adj is not None and not is_valid(best_adj, best_deg):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u in range(7):
            for v in range(7, 15):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        best_adj = adj
        best_deg = deg
        best_edges = count_edges(adj)
    
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([int(u), int(v)])
    
    return result
