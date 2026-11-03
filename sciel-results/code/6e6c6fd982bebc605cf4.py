import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Non-backtracking walk girth check from every vertex."""
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            q = deque([(s, -1, 0)])
            while q:
                u, prev, d = q.popleft()
                if d >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        return False
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        q.append((v, u, d + 1))
        return True

    def count_short_cycles(n, adj, g):
        """Count number of cycles shorter than g using non-backtracking walks."""
        count = 0
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            q = deque([(s, -1, 0)])
            while q:
                u, prev, d = q.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        count += 1
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        q.append((v, u, d + 1))
        return count

    def generate_simple_regular(n, k):
        """Generate a simple k-regular graph using configuration model with repair."""
        max_attempts = 50
        for attempt in range(max_attempts):
            stubs = list(range(n)) * k
            random.shuffle(stubs)
            
            edges = []
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u != v:
                    edges.append((u, v))
            
            edge_set = set()
            is_simple = True
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    is_simple = False
                    break
                edge_set.add(key)
            
            if is_simple and len(edges) == n * k // 2:
                return edges
            
            # Repair with 2-switches
            edge_dict = {}
            for i, (u, v) in enumerate(edges):
                key = (min(u, v), max(u, v))
                if key not in edge_dict:
                    edge_dict[key] = []
                edge_dict[key].append(i)
            
            for _ in range(n * k // 2):
                bad_key = None
                bad_idx1 = None
                for key, indices in edge_dict.items():
                    if len(indices) > 1:
                        bad_key = key
                        bad_idx1 = indices[0]
                        break
                if bad_key is None:
                    break
                
                u1, v1 = edges[bad_idx1]
                found_swap = False
                for j in random.sample(range(len(edges)), min(50, len(edges))):
                    if j == bad_idx1:
                        continue
                    u2, v2 = edges[j]
                    new_u1, new_v1 = u1, v2
                    new_u2, new_v2 = u2, v1
                    if new_u1 == new_v1 or new_u2 == new_v2:
                        continue
                    key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
                    key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
                    if key1 == key2:
                        continue
                    old_key1 = (min(u1, v1), max(u1, v1))
                    old_key2 = (min(u2, v2), max(u2, v2))
                    nc1 = len(edge_dict.get(key1, [])) - (1 if key1 == old_key1 else 0)
                    nc2 = len(edge_dict.get(key2, [])) - (1 if key2 == old_key2 else 0)
                    if nc1 < 0 or nc2 < 0:
                        continue
                    edges[bad_idx1] = (new_u1, new_v1)
                    edges[j] = (new_u2, new_v2)
                    if old_key1 in edge_dict:
                        edge_dict[old_key1].remove(bad_idx1)
                        if not edge_dict[old_key1]:
                            del edge_dict[old_key1]
                    if old_key2 in edge_dict:
                        edge_dict[old_key2].remove(j)
                        if not edge_dict[old_key2]:
                            del edge_dict[old_key2]
                    if key1 not in edge_dict:
                        edge_dict[key1] = []
                    edge_dict[key1].append(bad_idx1)
                    if key2 not in edge_dict:
                        edge_dict[key2] = []
                    edge_dict[key2].append(j)
                    found_swap = True
                    break
                if not found_swap:
                    break
            
            edge_set = set()
            is_simple = True
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    is_simple = False
                    break
                edge_set.add(key)
            
            if is_simple and len(edges) == n * k // 2:
                return edges
        
        return None

    def local_search_reduce_cycles(n, adj, edges, g, max_iters=500):
        """Local search to reduce number of short cycles using Simulated Annealing."""
        best_edges = list(edges)
        best_count = count_short_cycles(n, adj, g)
        current_edges = list(edges)
        current_count = best_count
        
        edge_count = {}
        for u, v in current_edges:
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1
            
        temp = 10.0
        cool_rate = 0.99
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            if current_count == 0:
                break
            
            i = random.randint(0, len(current_edges) - 1)
            u1, v1 = current_edges[i]
            old_key1 = (min(u1, v1), max(u1, v1))
            
            j = random.randint(0, len(current_edges) - 1)
            if j == i:
                continue
            u2, v2 = current_edges[j]
            old_key2 = (min(u2, v2), max(u2, v2))
            
            new_u1, new_v1 = u1, v2
            new_u2, new_v2 = u2, v1
            
            if new_u1 == new_v1 or new_u2 == new_v2:
                continue
            
            key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
            key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
            
            old_c1 = edge_count.get(old_key1, 0)
            old_c2 = edge_count.get(old_key2, 0)
            
            new_c1 = edge_count.get(key1, 0) - (1 if key1 == old_key1 else 0)
            new_c2 = edge_count.get(key2, 0) - (1 if key2 == old_key2 else 0)
            
            if new_c1 >= 1 or new_c2 >= 1 or key1 == key2:
                continue
            
            # Apply move
            current_edges[i] = (new_u1, new_v1)
            current_edges[j] = (new_u2, new_v2)
            
            edge_count[old_key1] = old_c1 - 1
            if edge_count[old_key1] <= 0:
                del edge_count[old_key1]
            edge_count[old_key2] = old_c2 - 1
            if edge_count[old_key2] <= 0:
                del edge_count[old_key2]
            
            if key1 not in edge_count:
                edge_count[key1] = 0
            edge_count[key1] += 1
            if key2 not in edge_count:
                edge_count[key2] = 0
            edge_count[key2] += 1
            
            new_adj = build_adj(current_edges, n)
            new_count = count_short_cycles(n, new_adj, g)
            
            delta = new_count - current_count
            
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_count = new_count
                adj = new_adj
                if current_count < best_count:
                    best_count = current_count
                    best_edges = list(current_edges)
            else:
                # Revert
                current_edges[i] = (u1, v1)
                current_edges[j] = (u2, v2)
                
                edge_count[key1] -= 1
                if edge_count[key1] <= 0:
                    del edge_count[key1]
                edge_count[key2] -= 1
                if edge_count[key2] <= 0:
                    del edge_count[key2]
                
                edge_count[old_key1] = edge_count.get(old_key1, 0) + 1
                edge_count[old_key2] = edge_count.get(old_key2, 0) + 1
            
            temp *= cool_rate
        
        return best_edges, best_count, adj

    # Seed construction (L-U-W style)
    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    seed_n, seed_edges = seed_construct(k, g)
    
    # Moore bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    best_n = seed_n
    best_edges = seed_edges
    
    # Try n values from Moore bound upward
    search_start = max(moore_bound, 430)
    search_end = min(seed_n - 10, 800) # Try up to 800
    
    for n in range(search_start, search_end + 1, 2):
        if time.time() > deadline - 20:
            break
        
        # Try 2 random graphs for this n
        for attempt in range(2):
            if time.time() > deadline - 20:
                break
            
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue
            
            adj = build_adj(edges, n)
            short_count = count_short_cycles(n, adj, g)
            
            if short_count == 0:
                if exact_girth_check(n, adj, g):
                    return n, edges
            
            edges, count, adj = local_search_reduce_cycles(n, adj, edges, g, max_iters=200)
            
            if count == 0:
                if exact_girth_check(n, adj, g):
                    return n, edges
    
    return seed_n, seed_edges
