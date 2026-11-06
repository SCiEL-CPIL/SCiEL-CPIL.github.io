import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    def is_triangle_free(adj, n):
        for i in range(n):
            a = adj[i]
            temp = a
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if adj[j] & a:
                    return False
                temp ^= lsb
        return True

    def has_independent_set_10_exact(adj, n, time_deadline):
        sys.setrecursionlimit(10000)
        
        def dfs(remaining, current_size):
            if time.time() > time_deadline:
                return None # Timeout
            if current_size >= 10:
                return True
            if current_size + remaining.bit_count() < 10:
                return False
            if remaining == 0:
                return False
            
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            new_remaining_include = remaining & ~adj[v] & ~lsb
            res = dfs(new_remaining_include, current_size + 1)
            if res is not None:
                return res
            
            # Branch 2: Exclude v
            new_remaining_exclude = remaining & ~lsb
            res = dfs(new_remaining_exclude, current_size)
            if res is not None:
                return res
            
            return False
        
        all_vertices = (1 << n) - 1
        return dfs(all_vertices, 0)

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                a ^= lsb
        return edges

    def greedy_independent_set_size(adj, n, limit=10):
        remaining = (1 << n) - 1
        size = 0
        while remaining and size < limit:
            min_deg = n
            min_v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (remaining & adj[v]).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                temp ^= lsb
            
            if min_v == -1:
                break
                
            size += 1
            remaining &= ~adj[min_v]
            remaining &= ~(1 << min_v)
            
        return size

    def search_sa(n, time_budget):
        end_time = start_time + time_budget
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.25:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove triangles
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    if (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j):
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
        
        prev_alpha = greedy_independent_set_size(adj, n, 10)
        best_adj = adj[:]
        best_alpha = prev_alpha
        
        T = 10.0
        cooling = 0.9995
        
        while time.time() < end_time:
            T *= cooling
            if T < 0.01:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            exists = (adj[u] >> v) & 1
            
            if exists:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                if (adj[u] & adj[v]) != 0:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            curr_alpha = greedy_independent_set_size(adj, n, 10)
            
            delta = curr_alpha - prev_alpha
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                prev_alpha = curr_alpha
                if curr_alpha < best_alpha:
                    best_alpha = curr_alpha
                    best_adj = adj[:]
            else:
                if exists:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return best_adj

    for n in [40, 39, 38, 37]:
        time_left = time_limit - (time.time() - start_time)
        if time_left < 10:
            break
        
        budget = min(time_left * 0.7, 20)
        adj = search_sa(n, budget)
        
        if adj is not None and is_triangle_free(adj, n):
            # Exact check
            deadline = time.time() + (time_limit - (time.time() - start_time)) * 0.2
            if deadline > time_limit:
                deadline = time_limit - 1.0
                
            result = has_independent_set_10_exact(adj, n, deadline)
            
            if result is False:
                edges = get_edges(adj, n)
                return {'n': n, 'edges': edges}

    # Fallback
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    edges = get_edges(adj, n)
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    result = construct()
    print(result['n'])
