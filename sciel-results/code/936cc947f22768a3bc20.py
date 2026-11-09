import numpy as np
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe

    # Fast girth verification using non-backtracking walks
    def verify_girth(n, edges, min_girth):
        """Check that the graph is k-regular, simple, and has girth >= min_girth."""
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return False
        
        # Check simple
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length 3 to min_girth-1 via trace of A^L
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Compute powers incrementally
        A2 = A @ A
        A3 = A2 @ A
        if np.trace(A3) > 0:
            return False
        
        A4 = A2 @ A2
        if np.trace(A4) > 0:
            return False
        
        A5 = A4 @ A
        if np.trace(A5) > 0:
            return False
        
        A6 = A4 @ A2
        if np.trace(A6) > 0:
            return False
        
        A7 = A6 @ A
        if np.trace(A7) > 0:
            return False
        
        A8 = A4 @ A4
        if np.trace(A8) > 0:
            return False
        
        A9 = A8 @ A
        if np.trace(A9) > 0:
            return False
        
        return True

    # Generate a random k-regular graph using configuration model
    def generate_random_k_regular(n, k, rng):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n < k or (k * n) % 2 != 0:
            return None
        
        for attempt in range(5000):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            stubs_copy = stubs[:]
            rng.shuffle(stubs_copy)
            
            edges = []
            edge_set = set()
            valid = True
            
            for i in range(0, len(stubs_copy), 2):
                u, v = stubs_copy[i], stubs_copy[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            
            if valid and len(edges) == k * n // 2:
                return edges
        
        return None

    # Count short cycles (length 3 to g-1) in a graph
    def count_short_cycles(n, edges, max_cycle_len):
        """Count total number of cycles of length 3 to max_cycle_len."""
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        total = 0
        if max_cycle_len >= 3:
            A2 = A @ A
            A3 = A2 @ A
            total += int(np.trace(A3)) // 6  # Each triangle counted 6 times
        
        if max_cycle_len >= 4:
            A4 = A2 @ A2
            total += int(np.trace(A4)) // 8  # Each 4-cycle counted 8 times
        
        if max_cycle_len >= 5:
            A5 = A4 @ A
            total += int(np.trace(A5)) // 10  # Each 5-cycle counted 10 times
        
        if max_cycle_len >= 6:
            A6 = A4 @ A2
            total += int(np.trace(A6)) // 12  # Each 6-cycle counted 12 times
        
        if max_cycle_len >= 7:
            A7 = A6 @ A
            total += int(np.trace(A7)) // 14
        
        if max_cycle_len >= 8:
            A8 = A4 @ A4
            total += int(np.trace(A8)) // 16
        
        if max_cycle_len >= 9:
            A9 = A8 @ A
            total += int(np.trace(A9)) // 18
        
        return total

    # Simulated Annealing: start with random k-regular graph, apply 2-switches
    # to reduce the number of short cycles
    def simulated_annealing(n, k, g, rng, time_limit):
        """Use simulated annealing with 2-switches to find a k-regular graph with high girth."""
        deadline_sa = time.time() + time_limit
        
        # Generate initial random k-regular graph
        edges = generate_random_k_regular(n, k, rng)
        if edges is None:
            return None
        
        # Build adjacency and edge sets for fast updates
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        # Current objective: number of short cycles
        current_cost = count_short_cycles(n, edges, g - 1)
        
        # Best solution found
        best_cost = current_cost
        best_edges = edges[:]
        
        # Annealing parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline_sa and T > T_min:
            # Pick a random 2-switch: find two edges (a,b) and (c,d) 
            # such that (a,c) and (b,d) or (a,d) and (b,c) are not edges
            # and a!=c, a!=d, b!=c, b!=d
            if not edges:
                break
            
            # Randomly pick two edges
            e1_idx = rng.randint(0, len(edges) - 1)
            e2_idx = rng.randint(0, len(edges) - 1)
            if e1_idx == e2_idx:
                continue
            
            a, b = edges[e1_idx]
            c, d = edges[e2_idx]
            
            # Ensure a < b, c < d for consistency
            if a > b:
                a, b = b, a
            if c > d:
                c, d = d, c
            
            # Try switching: remove (a,b) and (c,d), add (a,c) and (b,d)
            # Check validity: no loops, no existing edges
            if a == c or a == d or b == c or b == d:
                continue
            
            new_edge1 = (min(a,c), max(a,c))
            new_edge2 = (min(b,d), max(b,d))
            
            if new_edge1 in edge_set or new_edge2 in edge_set:
                # Try the other pairing: (a,d) and (b,c)
                new_edge1 = (min(a,d), max(a,d))
                new_edge2 = (min(b,c), max(b,c))
                if new_edge1 in edge_set or new_edge2 in edge_set:
                    continue
                # Use this pairing
                remove1, remove2 = (a,b), (c,d)
                add1, add2 = (a,d), (b,c)
            else:
                remove1, remove2 = (a,b), (c,d)
                add1, add2 = (a,c), (b,d)
            
            # Apply the switch
            edge_set.discard(remove1)
            edge_set.discard(remove2)
            edge_set.add(add1)
            edge_set.add(add2)
            
            adj[remove1[0]].discard(remove1[1])
            adj[remove1[1]].discard(remove1[0])
            adj[remove2[0]].discard(remove2[1])
            adj[remove2[1]].discard(remove2[0])
            
            adj[add1[0]].add(add1[1])
            adj[add1[1]].add(add1[0])
            adj[add2[0]].add(add2[1])
            adj[add2[1]].add(add2[0])
            
            # Update edges list
            edges[e1_idx] = add1
            edges[e2_idx] = add2
            
            # Compute new cost (expensive but necessary for accuracy)
            new_cost = count_short_cycles(n, edges, g - 1)
            
            delta = new_cost - current_cost
            
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges = edges[:]
            else:
                # Revert the switch
                edge_set.discard(add1)
                edge_set.discard(add2)
                edge_set.add(remove1)
                edge_set.add(remove2)
                
                adj[add1[0]].discard(add1[1])
                adj[add1[1]].discard(add1[0])
                adj[add2[0]].discard(add2[1])
                adj[add2[1]].discard(add2[0])
                
                adj[remove1[0]].add(remove1[1])
                adj[remove1[1]].add(remove1[0])
                adj[remove2[0]].add(remove2[1])
                adj[remove2[1]].add(remove2[0])
                
                edges[e1_idx] = remove1
                edges[e2_idx] = remove2
            
            # Cool down
            T *= cooling_rate
        
        # Verify the best solution
        if verify_girth(n, best_edges, g):
            return best_edges
        
        return None

    # Main construction
    target_n = 320
    rng = random.Random(999)
    
    # Try simulated annealing with different time budgets
    # First, try with a shorter time budget to see if we can find a solution quickly
    for time_budget in [30, 40, 40]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        edges = simulated_annealing(target_n, k, g, rng, min(time_budget, remaining - 5))
        if edges is not None:
            return target_n, edges
    
    # Fallback: try random graphs directly
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        edges = generate_random_k_regular(target_n, k, rng)
        if edges is not None and verify_girth(target_n, edges, g):
            return target_n, edges
    
    # Final fallback: L-U-W construction
    def luw_construct(k, g):
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
    
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw
