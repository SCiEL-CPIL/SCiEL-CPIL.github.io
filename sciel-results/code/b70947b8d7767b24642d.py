import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for final check/return
    
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

    # Correct Girth Checker using non-backtracking walks
    def get_girth(adj, min_girth=3):
        n = len(adj)
        # adj is a list of lists
        # Convert to set for faster lookup
        adj_set = [set(a) for a in adj]
        
        # Use non-backtracking walks from each vertex
        # State: (current_vertex, previous_vertex, length)
        from collections import deque
        
        best = float('inf')
        
        for start in range(n):
            # BFS in the non-backtracking state space
            # State: (u, v) means we're at u, came from v
            visited = set()
            queue = deque()
            
            # Initialize: from start, go to each neighbor
            for v in adj[start]:
                state = (v, start)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, start, 1))
            
            while queue:
                u, prev, length = queue.popleft()
                
                # If we've gone too far, stop
                if length >= best:
                    continue
                
                for v in adj[u]:
                    if v == prev:
                        continue  # No backtracking
                    
                    new_state = (v, u)
                    new_length = length + 1
                    
                    if v == start:
                        # Found a cycle
                        if new_length < best:
                            best = new_length
                        continue
                    
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((v, u, new_length))
            
            if best < min_girth:
                break
        
        return best if best != float('inf') else 2 * n

    # Better Random K-Regular Graph Generator
    def gen_random_k_reg(n, k, max_attempts=100):
        """Generate a random k-regular simple graph using configuration model with repair."""
        for attempt in range(max_attempts):
            # Create stubs
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            # Pair stubs
            adj = [[] for _ in range(n)]
            valid = True
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid and all(len(a) == k for a in adj):
                return adj
        
        return None

    # Improved Optimization Loop
    def optimize(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if not adj:
            return None
        
        # Convert to sets for faster lookup
        adj_set = [set(a) for a in adj]
        
        # Penalty: weighted sum of short cycles
        def calc_penalty(adj_set):
            pen = 0.0
            # Count triangles
            tri_count = 0
            for u in range(n):
                for v in adj_set[u]:
                    if u < v:
                        common = adj_set[u] & adj_set[v]
                        tri_count += len(common)
            pen += tri_count * 10.0  # Weight triangles heavily
            
            # Count 4-cycles (approximate)
            four_count = 0
            for u in range(n):
                for v in adj_set[u]:
                    if u < v:
                        common = adj_set[u] & adj_set[v]
                        for w in common:
                            # Check if there's a 4-cycle u-v-w-?-u
                            # This is expensive, so skip for now
                            pass
            # pen += four_count * 5.0
            
            return pen
        
        current_penalty = calc_penalty(adj_set)
        
        end_time = time.time() + time_limit
        best_adj = None
        best_penalty = current_penalty
        
        # Simulated annealing with 2-switches
        temperature = 1.0
        cooling_rate = 0.999
        
        while time.time() < end_time:
            # 2-Switch
            u = random.randint(0, n-1)
            if len(adj[u]) == 0:
                continue
            v = random.choice(adj[u])
            w = random.randint(0, n-1)
            if w == u or w == v:
                continue
            if len(adj[w]) == 0:
                continue
            x = random.choice(adj[w])
            if x == u or x == v:
                continue
            
            # Check validity
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
            
            new_penalty = calc_penalty(adj_set)
            
            # Acceptance criterion
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_adj = [list(a) for a in adj]
                
                # Check if girth is sufficient
                if current_penalty < 1.0:
                    girth = get_girth(adj, min_girth=g)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return (n, edges)
            else:
                # Simulated annealing: accept with probability exp(-delta/T)
                delta = new_penalty - current_penalty
                if random.random() < math.exp(-delta / temperature):
                    current_penalty = new_penalty
                else:
                    # Reject: undo swap
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
            
            # Cool down
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
        
        # Return best found if it meets girth requirement
        if best_adj is not None:
            girth = get_girth(best_adj, min_girth=g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in best_adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
        
        return None

    # Try several N values, starting from smaller
    # Moore bound for k=5, g=9: 2*(1+5+5*4+5*4^2+5*4^3+5*4^4)/2 = 1 + 5 + 20 + 80 + 320 + 1280 = 1706
    # But we want smaller, so try values around 428-800
    candidates = [500, 600, 700, 800]
    
    for n in candidates:
        if time.time() > deadline - 15:
            break
        time_budget = min(25, deadline - time.time() - 10)
        if time_budget < 5:
            break
            
        result = optimize(n, k, g, time_budget)
        if result:
            return result

    # Fallback
    return luw_construction()
