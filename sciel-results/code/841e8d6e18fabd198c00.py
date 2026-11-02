"""De Bruijn-type finite-state construction for regular graphs of large girth, enhanced with local search."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    def luw_construct(k, g):
        """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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

    # Fallback result
    best_n, best_edges = luw_construct(k, g)

    if k == 3 and g == 16:
        # Moore bound for (3, 16) is 510, so n must be even and >= 512.
        # We try to find a smaller graph using local search.
        
        def generate_random_3_reg(n):
            """Generate a random 3-regular graph on n vertices."""
            if n < 4 or n % 2 != 0:
                return None, None
            
            # Configuration model with rejection
            # To ensure simplicity, we use a specific algorithm:
            # 1. Create list of stubs [0,0,0, 1,1,1, ...]
            # 2. Shuffle
            # 3. Pair up
            # 4. Check for loops and multi-edges. If any, restart (up to limit).
            # For n > 100, restarts are rare.
            
            max_attempts = 50
            for _ in range(max_attempts):
                stubs = [i for i in range(n) for _ in range(3)]
                random.shuffle(stubs)
                
                adj = [[] for _ in range(n)]
                edge_set = set()
                valid = True
                for i in range(len(stubs) // 2):
                    u, v = stubs[2*i], stubs[2*i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                    adj[u].append(v)
                    adj[v].append(u)
                
                if valid:
                    # Verify degrees
                    if all(len(a) == 3 for a in adj):
                        edges = [tuple(sorted(e)) for e in edge_set]
                        return adj, edges
            return None, None

        def calc_girth(adj, n):
            """Calculate the girth of the graph. Returns infinity if no cycle (tree, impossible for n>1 3-reg)."""
            min_girth = float('inf')
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= min_girth // 2:
                        continue
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Cycle found
                            cycle_len = du + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
            return min_girth

        def try_improve(n, time_budget):
            """Attempt to find a 3-regular graph on n vertices with girth >= 16 using 2-switch local search."""
            adj, edges = generate_random_3_reg(n)
            if adj is None:
                return False, None, None
            
            current_girth = calc_girth(adj, n)
            if current_girth >= 16:
                return True, n, [tuple(e) for e in edges]

            end_time = time.time() + time_budget
            edge_list = list(edges)
            
            while time.time() < end_time:
                # Pick two random edges
                if len(edge_list) < 2:
                    break
                e1_idx = random.randint(0, len(edge_list) - 1)
                e2_idx = random.randint(0, len(edge_list) - 1)
                if e1_idx == e2_idx:
                    continue
                
                u1, v1 = edge_list[e1_idx]
                u2, v2 = edge_list[e2_idx]
                
                # Check if edges are disjoint (no shared vertices)
                if len({u1, v1, u2, v2}) < 4:
                    continue
                
                # Try swap 1: (u1, u2) and (v1, v2)
                new_e1 = (min(u1, u2), max(u1, u2))
                new_e2 = (min(v1, v2), max(v1, v2))
                
                # Check if new edges already exist
                if new_e1 in edge_set or new_e2 in edge_set:
                    continue
                if new_e1 == new_e2:
                    continue
                
                # Perform swap
                edge_set.remove((u1, v1))
                edge_set.remove((u2, v2))
                edge_set.add(new_e1)
                edge_set.add(new_e2)
                
                # Update adjacency
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].append(u2)
                adj[u2].append(u1)
                adj[v1].append(v2)
                adj[v2].append(v1)
                
                edge_list[e1_idx] = new_e1
                edge_list[e2_idx] = new_e2
                
                # Calculate new girth
                new_girth = calc_girth(adj, n)
                
                if new_girth < current_girth:
                    # Revert
                    edge_set.remove(new_e1)
                    edge_set.remove(new_e2)
                    edge_set.add((u1, v1))
                    edge_set.add((u2, v2))
                    
                    adj[u1].remove(u2)
                    adj[u2].remove(u1)
                    adj[v1].remove(v2)
                    adj[v2].remove(v1)
                    
                    adj[u1].append(v1)
                    adj[v1].append(u1)
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    
                    edge_list[e1_idx] = (u1, v1)
                    edge_list[e2_idx] = (u2, v2)
                else:
                    # Accept
                    current_girth = new_girth
                    if current_girth >= 16:
                        return True, n, [tuple(e) for e in edge_set]
            return False, None, None

        # Search for smaller n
        # Start from Moore bound 512
        # We allocate time based on n. Smaller n are harder, give them more time?
        # Or just iterate n and give each a small budget.
        # Let's try n from 512 to 800.
        for n in range(512, 802, 2):
            if time.time() > deadline - 15:
                break
            
            # Budget per n: 5 seconds
            found, res_n, res_edges = try_improve(n, 5.0)
            if found:
                return res_n, res_edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
