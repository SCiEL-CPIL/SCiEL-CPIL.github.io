import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    # Fallback: LUW construction from seed
    def luw_construction():
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

    # Fast girth checker using BFS in non-backtracking state space
    def get_girth(adj, min_girth=3):
        n = len(adj)
        adj_set = [set(a) for a in adj]
        best = float('inf')
        
        for start in range(n):
            # BFS in non-backtracking state space
            # State: (u, prev) means at u, came from prev
            visited = set()
            queue = deque()
            
            for v in adj[start]:
                state = (v, start)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, start, 1))
            
            while queue:
                u, prev, length = queue.popleft()
                if length >= best:
                    continue
                
                for v in adj[u]:
                    if v == prev:
                        continue
                    
                    new_length = length + 1
                    if v == start:
                        if new_length < best:
                            best = new_length
                        continue
                    
                    new_state = (v, u)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((v, u, new_length))
            
            if best < min_girth:
                break
        
        return best if best != float('inf') else 2 * n

    # Generate random k-regular graph using configuration model with repair
    def gen_random_k_reg(n, k, max_attempts=50):
        for attempt in range(max_attempts):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            adj_set = [set() for _ in range(n)]
            valid = True
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj_set[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
            
            if valid and all(len(a) == k for a in adj):
                return adj, adj_set
        
        return None, None

    # Count short cycles (up to length g-1) for penalty
    def count_short_cycles(adj, adj_set, n, max_len):
        count = 0
        # Count cycles of length 3 to max_len using BFS
        for start in range(n):
            visited = set()
            queue = deque()
            for v in adj[start]:
                visited.add((v, start))
                queue.append((v, start, 1))
            
            while queue:
                u, prev, length = queue.popleft()
                if length >= max_len:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        count += 1
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, length + 1))
        return count

    # Optimize using simulated annealing with 2-switches
    def optimize(n, k, g, time_limit):
        adj, adj_set = gen_random_k_reg(n, k)
        if not adj:
            return None
        
        current_penalty = count_short_cycles(adj, adj_set, n, g)
        end_time = time.time() + time_limit
        best_adj = None
        best_penalty = current_penalty
        temperature = 1.0
        cooling_rate = 0.9995
        
        while time.time() < end_time:
            # Pick two random edges to swap
            u = random.randint(0, n-1)
            if not adj[u]:
                continue
            v = random.choice(adj[u])
            
            w = random.randint(0, n-1)
            if w == u or w == v or not adj[w]:
                continue
            x = random.choice(adj[w])
            if x == u or x == v or x == w:
                continue
            
            # Check if swap is valid (no multi-edges)
            if w in adj_set[u] or x in adj_set[v] or w == x:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            adj_set[w].discard(x)
            adj_set[x].discard(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            adj_set[u].add(w)
            adj_set[w].add(u)
            adj_set[v].add(x)
            adj_set[x].add(v)
            
            new_penalty = count_short_cycles(adj, adj_set, n, g)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_adj = [list(a) for a in adj]
                
                if current_penalty == 0:
                    girth = get_girth(adj, min_girth=g)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return (n, edges)
            else:
                delta = new_penalty - current_penalty
                if random.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_penalty = new_penalty
                else:
                    # Undo swap
                    adj[u].remove(w)
                    adj[w].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    
                    adj_set[u].discard(w)
                    adj_set[w].discard(u)
                    adj_set[v].discard(x)
                    adj_set[x].discard(v)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[w].append(x)
                    adj[x].append(w)
                    
                    adj_set[u].add(v)
                    adj_set[v].add(u)
                    adj_set[w].add(x)
                    adj_set[x].add(w)
            
            temperature *= cooling_rate
        
        if best_adj is not None and best_penalty == 0:
            girth = get_girth(best_adj, min_girth=g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in best_adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
        
        return None

    # Try to find a solution with random optimization, then fallback to LUW
    # Moore bound for k=5, g=9 is around 428, try values in reasonable range
    candidates = [450, 500, 550, 600]
    
    for n in candidates:
        if time.time() > deadline - 20:
            break
        time_budget = min(20, deadline - time.time() - 10)
        if time_budget < 5:
            break
        
        result = optimize(n, k, g, time_budget)
        if result:
            return result

    # Fallback to LUW construction
    return luw_construction()
