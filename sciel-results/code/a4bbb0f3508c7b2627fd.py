import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def count_edges_in_set(mask, adj):
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            count += (adj[u] & mask).bit_count()
        return count // 2

    def count_triangles_in_set(mask, adj):
        if mask == 0:
            return 0
        total = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            total += count_edges_in_set(adj[u] & mask, adj)
        return total // 3

    def count_i5s_in_set(mask, adj):
        if mask == 0:
            return 0
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        n_v = len(vertices)
        if n_v < 5:
            return 0
        count = 0
        for i in range(n_v):
            for j in range(i+1, n_v):
                u, v = vertices[i], vertices[j]
                if not ((adj[u] >> v) & 1):
                    common_non = mask & ~(adj[u] | adj[v])
                    common_non &= ~(1 << u)
                    common_non &= ~(1 << v)
                    count += count_triangles_in_set(common_non, adj)
        return count // 5

    def count_i6s(n, adj):
        all_mask = (1 << n) - 1
        total = 0
        for u in range(n):
            non_n = all_mask & ~(adj[u] | (1 << u))
            total += count_i5s_in_set(non_n, adj)
        return total // 6

    def calculate_cost(n, adj):
        k4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    k4_count += count_edges_in_set(common, adj)
        k4_count //= 4
        i6_count = count_i6s(n, adj)
        return k4_count + i6_count

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def turan_graph_3(n):
        adj = [0] * n
        sizes = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
        parts = []
        idx = 0
        for s in sizes:
            parts.append(list(range(idx, idx + s)))
            idx += s
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def try_local_search(n, adj, time_limit):
        current_cost = calculate_cost(n, adj)
        if current_cost == 0:
            return adj, 0
        
        all_mask = (1 << n) - 1
        T = 5.0
        T_min = 0.05
        cooling = 0.995
        
        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_edge = bool((adj[u] >> v) & 1)
            
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            non_common = all_mask & ~(adj[u] | adj[v])
            non_common &= ~(1 << u)
            non_common &= ~(1 << v)
            
            tri_delta = count_triangles_in_set(common, adj)
            i5_delta = count_i5s_in_set(non_common, adj)
            
            if is_edge:
                delta = -tri_delta + i5_delta
            else:
                delta = tri_delta - i5_delta
            
            if delta < 0:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_cost += delta
                if current_cost == 0:
                    return adj, 0
            else:
                p = math.exp(-delta / T)
                if random.random() < p:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_cost += delta
                    if current_cost == 0:
                        return adj, 0
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return adj, current_cost

    best_graph = None
    best_n = 0

    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 15:
                break
            
            adj = turan_graph_3(target_n)
            
            current_cost = calculate_cost(target_n, adj)
            if current_cost == 0:
                best_graph = adj
                best_n = target_n
                found = True
                break
            
            time_left = deadline - 15 - time.time()
            if time_left < 5:
                break
            
            per_restart = min(time_left / max(1, 20 - restart), 10.0)
            adj, current_cost = try_local_search(target_n, adj, time.time() + per_restart)
            
            if current_cost == 0:
                best_graph = adj
                best_n = target_n
                found = True
                break
        
        if found:
            break

    if best_graph is not None:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
    
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': get_edges(adj, 17)}
