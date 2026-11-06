import random
import time

def construct():
    start_time = time.time()
    timeout = 145.0 # 150s limit, leave 5s
    
    n = 40
    m_target = 401
    
    def get_book_size(adj, u, v):
        common = adj[u] & adj[v]
        return bin(common).count('1')
        
    def calculate_max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            b = get_book_size(adj, u, v)
            if b > max_b:
                max_b = b
        return max_b

    def calculate_sum_books(adj, edges):
        sum_b = 0
        for u, v in edges:
            sum_b += get_book_size(adj, u, v)
        return sum_b

    def add_edge(adj, edges, edge_set, u, v):
        if u > v: u, v = v, u
        if (u, v) in edge_set: return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
        edge_set.add((u, v))
        return True

    def remove_edge(adj, edges, edge_set, u, v):
        if u > v: u, v = v, u
        if (u, v) not in edge_set: return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Find and remove from list
        for i, e in enumerate(edges):
            if e == (u, v):
                edges.pop(i)
                break
        edge_set.remove((u, v))
        return True

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_graph = None
    best_score = float('inf')
    
    # Phase 1: Greedy Construction
    # Try to construct graph with max book <= B
    for B in range(7, 15):
        if time.time() - start_time > timeout: break
        
        adj = [0] * n
        edges = []
        edge_set = set()
        
        random.shuffle(all_edges)
        
        added = 0
        for u, v in all_edges:
            if added == m_target:
                break
            
            common = adj[u] & adj[v]
            common_count = bin(common).count('1')
            
            if common_count > B:
                continue
            
            valid = True
            w = 0
            temp_common = common
            while temp_common:
                if temp_common & 1:
                    # w is a common neighbor
                    buw = bin(adj[u] & adj[w]).count('1')
                    if buw + 1 > B:
                        valid = False
                        break
                    bvw = bin(adj[v] & adj[w]).count('1')
                    if bvw + 1 > B:
                        valid = False
                        break
                temp_common >>= 1
                w += 1
            
            if valid:
                add_edge(adj, edges, edge_set, u, v)
                added += 1
        
        if added == m_target:
            actual_max = calculate_max_book(adj, edges)
            if actual_max < best_score:
                best_score = actual_max
                best_graph = (adj[:], edges[:], edge_set.copy())
            
            if B == 7:
                break

    if best_graph is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    # Phase 2: Local Search
    adj = best_graph[0]
    edges = best_graph[1]
    edge_set = best_graph[2]
    
    current_max = best_score
    current_sum = calculate_sum_books(adj, edges)
    
    best_max = current_max
    best_sum = current_sum
    best_adj = adj[:]
    best_edges = edges[:]
    best_edge_set = edge_set.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if len(edges) > 0:
            e_idx = random.randint(0, len(edges)-1)
            u1, v1 = edges[e_idx]
            
            added = False
            for _ in range(5):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                if u2 > v2: u2, v2 = v2, u2
                if (u2, v2) in edge_set: continue
                if u1 > v1: u1c, v1c = v1, u1
                else: u1c, v1c = u1, v1
                if (u2, v2) == (u1c, v1c): continue
                
                old_adj = adj[:]
                old_edges = edges[:]
                old_edge_set = edge_set.copy()
                
                remove_edge(adj, edges, edge_set, u1, v1)
                add_edge(adj, edges, edge_set, u2, v2)
                
                new_max = calculate_max_book(adj, edges)
                new_sum = calculate_sum_books(adj, edges)
                
                if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                    best_max = new_max
                    best_sum = new_sum
                    best_adj = adj[:]
                    best_edges = edges[:]
                    best_edge_set = edge_set.copy()
                    current_max = new_max
                    current_sum = new_sum
                elif new_max < current_max or (new_max == current_max and new_sum < current_sum):
                    current_max = new_max
                    current_sum = new_sum
                else:
                    delta = (new_max - current_max) * 10 + (new_sum - current_sum)
                    if delta > 0:
                        if random.random() < pow(2.718, -delta / T):
                            current_max = new_max
                            current_sum = new_sum
                        else:
                            adj = old_adj
                            edges = old_edges
                            edge_set = old_edge_set
                    else:
                        current_max = new_max
                        current_sum = new_sum
                
                added = True
                break
            
            if added:
                T *= cooling
                if T < T_min:
                    T = T_min
    
    return [[u, v] for u, v in best_edges]
