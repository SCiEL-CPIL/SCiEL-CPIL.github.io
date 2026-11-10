import time
import random
import sys
from collections import deque
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    
    Strategy:
    1. For k=4, g=10, attempt to find the 320-vertex cage using randomized local search.
       - Uses a fast heuristic energy function based on random BFS sampling to estimate girth.
       - Performs edge swaps to minimize short cycles.
       - Verifies the final candidate with a full girth check.
    2. If that fails within time, try larger n (up to a limit) with the same search.
    3. Fall back to the Lazebnik-Ustimenko-Woldar construction which is guaranteed to be valid.
    """
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin

    def elapsed():
        return time.time() - t0

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        
        def generate_random_4_reg_bipartite(n):
            """
            Generates a simple 4-regular bipartite graph on n vertices (n/2 in each part).
            Uses configuration model with multi-edge removal.
            """
            if n % 2 != 0:
                return None
            
            half_n = n // 2
            adj_list = [[] for _ in range(n)]
            
            left_stubs = [i for i in range(half_n) for _ in range(4)]
            right_stubs = [half_n + i for i in range(half_n) for _ in range(4)]
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            for l, r in zip(left_stubs, right_stubs):
                adj_list[l].append(r)
                adj_list[r].append(l)
            
            def has_multi_edges(adj_list):
                for v in range(n):
                    if len(adj_list[v]) != len(set(adj_list[v])):
                        return True
                return False
            
            max_iters = 5000
            for _ in range(max_iters):
                if not has_multi_edges(adj_list):
                    break
                
                v = -1
                for i in range(n):
                    if len(adj_list[i]) != len(set(adj_list[i])):
                        v = i
                        break
                if v == -1:
                    break
                
                counts = {}
                for u in adj_list[v]:
                    counts[u] = counts.get(u, 0) + 1
                u = [key for key, val in counts.items() if val > 1][0]
                
                found_swap = False
                for _ in range(200):
                    v_prime = random.randint(0, n - 1)
                    if v_prime == v:
                        continue
                    if v_prime < half_n and v < half_n:
                        continue
                    if v_prime >= half_n and v >= half_n:
                        continue
                    
                    if v_prime < half_n:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x >= half_n]
                    else:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x < half_n]
                        
                    if not u_prime_candidates:
                        continue
                        
                    u_prime = random.choice(u_prime_candidates)
                    
                    if u_prime in adj_list[v]:
                        continue
                    if u in adj_list[v_prime]:
                        continue
                    
                    if v not in adj_list[v] or u_prime not in adj_list[v_prime]:
                        continue
                    if v not in adj_list[u] or u_prime not in adj_list[v_prime]:
                         pass # Already checked

                    # Perform swap
                    adj_list[v].remove(u)
                    adj_list[v].append(u_prime)
                    
                    adj_list[u].remove(v)
                    adj_list[u].append(v_prime)
                    
                    adj_list[v_prime].remove(u_prime)
                    adj_list[v_prime].append(u)
                    
                    adj_list[u_prime].remove(v_prime)
                    adj_list[u_prime].append(v)
                    
                    found_swap = True
                    break
                
                if not found_swap:
                    continue
            
            if not has_multi_edges(adj_list):
                return adj_list
            return None

        def fast_girth_estimate(adj_list, sample_size=50):
            """
            Estimates girth by performing BFS from a random sample of vertices.
            Returns a continuous energy score: lower is better.
            Uses soft-min over short cycles found to provide gradient signal.
            """
            n_vertices = len(adj_list)
            cycle_lengths = []
            sample_nodes = random.sample(range(n_vertices), min(sample_size, n_vertices))
            
            for start in sample_nodes:
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] > g // 2:
                        continue
                    for v in adj_list[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                cycle_lengths.append(cycle_len)
                                if cycle_len <= 4:
                                    return 1.0 / cycle_len  # Very bad, high energy
            if not cycle_lengths:
                return 0.0  # No short cycles found, perfect
            # Use soft-min: sum of exp(-alpha * (g - L)) for each cycle of length L
            # This gives a smooth penalty that increases as cycles get shorter
            alpha = 0.5
            energy = 0.0
            for L in cycle_lengths:
                energy += math.exp(-alpha * (g - L))
            return energy

        def exact_girth_check(adj_list):
            """
            Exact girth check using BFS from all vertices.
            Used only for final verification.
            """
            n_vertices = len(adj_list)
            min_girth = 100
            
            for start in range(n_vertices):
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= g // 2:
                        continue
                    for v in adj_list[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth <= 4:
                                    return min_girth
            return min_girth

        def try_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices with girth >= 10.
            Uses simulated annealing with continuous energy landscape.
            """
            adj = generate_random_4_reg_bipartite(n)
            if adj is None:
                return None
            
            # Initial energy check
            current_energy = fast_girth_estimate(adj, sample_size=50)
            if current_energy == 0.0:
                # Verify exactly
                if exact_girth_check(adj) >= g:
                    edges = []
                    for u in range(n // 2):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges

            # Local search with simulated annealing-like acceptance
            temperature = 0.5
            cooling_rate = 0.9995
            best_adj = [lst[:] for lst in adj]
            best_energy = current_energy
            steps = 0
            no_improve_steps = 0
            max_no_improve = 5000
            
            while time.time() - t0 < time_limit:
                steps += 1
                # Pick two random edges (u, v) and (u', v')
                u = random.randint(0, n // 2 - 1)
                if not adj[u]:
                    continue
                v = random.choice(adj[u])
                
                u_prime = random.randint(0, n // 2 - 1)
                if u_prime == u:
                    continue
                if not adj[u_prime]:
                    continue
                v_prime = random.choice(adj[u_prime])
                
                if v_prime in adj[u]:
                    continue
                if v in adj[u_prime]:
                    continue
                
                if v not in adj[u] or v_prime not in adj[u_prime]:
                    continue
                if u not in adj[v] or u_prime not in adj[v_prime]:
                    continue

                # Perform swap
                adj[u].remove(v)
                adj[u].append(v_prime)
                adj[v].remove(u)
                adj[v].append(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                
                new_energy = fast_girth_estimate(adj, sample_size=20)
                
                # Accept if better, or with probability exp(-delta/T)
                delta = new_energy - current_energy
                if delta < 0 or (temperature > 0.01 and random.random() < math.exp(-delta / temperature)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj = [lst[:] for lst in adj]
                        no_improve_steps = 0
                        
                        # If we have zero energy (no short cycles found), verify exactly
                        if best_energy == 0.0:
                            if exact_girth_check(adj) >= g:
                                edges = []
                                for u_node in range(n // 2):
                                    for v_node in adj[u_node]:
                                        if u_node < v_node:
                                            edges.append((u_node, v_node))
                                return n, edges
                    else:
                        no_improve_steps += 1
                else:
                    # Revert
                    adj[u].remove(v_prime)
                    adj[u].append(v)
                    adj[v].remove(u_prime)
                    adj[v].append(u)
                    
                    adj[u_prime].remove(v)
                    adj[u_prime].append(v_prime)
                    adj[v_prime].remove(u)
                    adj[v_prime].append(u_prime)
                    no_improve_steps += 1
                    
                temperature *= cooling_rate
                if temperature < 0.01:
                    temperature = 0.01
                
                # Restart if stuck
                if no_improve_steps > max_no_improve:
                    no_improve_steps = 0
                    temperature = max(temperature, 0.1)
                    # Try to escape local minimum by accepting worse moves more readily

            # Final check on best_adj
            if best_energy == 0.0:
                if exact_girth_check(best_adj) >= g:
                    edges = []
                    for u_node in range(n // 2):
                        for v_node in best_adj[u_node]:
                            if u_node < v_node:
                                edges.append((u_node, v_node))
                    return n, edges

            return None

        # Try specific n values
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400, 500]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Allocate time
            if n == 320:
                limit = min(60, DEADLINE - (time.time() - t0) - 5)
            else:
                limit = min(20, DEADLINE - (time.time() - t0) - 5)
                
            if limit < 1:
                continue
                
            try:
                result = try_local_search(n, limit)
                if result:
                    return result
            except Exception:
                continue

    # Fallback: L-U-W Construction
    q = k
    if g % 2 == 1:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = max(4, g - 4)
        if d % 2 == 1:
            d += 1
    
    if q not in [2, 3, 4, 5, 7, 11, 13]:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        q = 4
        while not (is_prime(q) or q == 4):
            q += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
