import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Strategy:
    1. For k=4, g=10, attempt to find the 320-vertex cage using randomized local search.
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
            # Left: 0..half_n-1, Right: half_n..n-1
            
            # Use lists for adjacency to allow O(1) removal by value
            # adj_list[v] is a list of 4 neighbors
            adj_list = [[] for _ in range(n)]
            
            # Create stubs
            left_stubs = [i for i in range(half_n) for _ in range(4)]
            right_stubs = [half_n + i for i in range(half_n) for _ in range(4)]
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            # Pair them up
            for l, r in zip(left_stubs, right_stubs):
                adj_list[l].append(r)
                adj_list[r].append(l)
            
            # Remove multi-edges by swapping
            def has_multi_edges(adj_list):
                for v in range(n):
                    if len(adj_list[v]) != len(set(adj_list[v])):
                        return True
                return False
            
            max_iters = 2000
            for _ in range(max_iters):
                if not has_multi_edges(adj_list):
                    break
                
                # Find a vertex with duplicate
                v = -1
                for i in range(n):
                    if len(adj_list[i]) != len(set(adj_list[i])):
                        v = i
                        break
                if v == -1:
                    break
                
                # Find a duplicate neighbor u
                counts = {}
                for u in adj_list[v]:
                    counts[u] = counts.get(u, 0) + 1
                u = [key for key, val in counts.items() if val > 1][0]
                
                # Find a vertex v' != v that is connected to some u'
                # We want to swap (v, u) with (v', u')
                # Conditions:
                # 1. u' != u
                # 2. (v, u') is not an edge
                # 3. (v', u) is not an edge
                # 4. v' and u' are in opposite parts
                
                found_swap = False
                # Try random v'
                for _ in range(200):
                    v_prime = random.randint(0, n - 1)
                    if v_prime == v:
                        continue
                    if v_prime < half_n and v < half_n: # Both left? No, bipartite
                        continue
                    if v_prime >= half_n and v >= half_n:
                        continue
                    
                    # Pick a neighbor u' of v_prime
                    # We need u' to be in the other part
                    if v_prime < half_n:
                        # v' is Left, neighbors are Right
                        u_prime_candidates = [x for x in adj_list[v_prime] if x >= half_n]
                    else:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x < half_n]
                        
                    if not u_prime_candidates:
                        continue
                        
                    u_prime = random.choice(u_prime_candidates)
                    
                    # Check if swap is valid
                    # We need to remove (v, u) and (v', u')
                    # And add (v, u') and (v', u)
                    
                    # (v, u') must not exist in adj_list[v]
                    if u_prime in adj_list[v]:
                        continue
                    # (v', u) must not exist in adj_list[v']
                    if u in adj_list[v_prime]:
                        continue
                    
                    # Perform swap
                    # Remove u from v
                    if u not in adj_list[v]:
                        continue
                    adj_list[v].remove(u)
                    # Add u' to v
                    adj_list[v].append(u_prime)
                    
                    # Remove v from u
                    if v not in adj_list[u]:
                        continue
                    adj_list[u].remove(v)
                    # Add v' to u
                    adj_list[u].append(v_prime)
                    
                    # Remove u' from v'
                    if u_prime not in adj_list[v_prime]:
                        continue
                    adj_list[v_prime].remove(u_prime)
                    # Add u to v'
                    adj_list[v_prime].append(u)
                    
                    # Remove v' from u'
                    if v_prime not in adj_list[u_prime]:
                        continue
                    adj_list[u_prime].remove(v_prime)
                    # Add v to u'
                    adj_list[u_prime].append(v)
                    
                    found_swap = True
                    break
                
                if not found_swap:
                    # If we can't fix it with random swaps, try another vertex or give up
                    continue
            
            if not has_multi_edges(adj_list):
                return adj_list
            return None

        def get_girth(adj_list):
            """
            Returns the length of the shortest cycle.
            Uses BFS from each vertex.
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
                    for v in adj_list[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth <= 4:
                                    return min_girth
            return min_girth

        def try_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices with girth >= 10.
            """
            # Generate initial graph
            adj = generate_random_4_reg_bipartite(n)
            if adj is None:
                return None
            
            current_girth = get_girth(adj)
            
            if current_girth >= 10:
                # Success
                edges = []
                for u in range(n // 2):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

            # Local search
            while time.time() - t0 < time_limit:
                # Pick two random edges (u, v) and (u', v')
                # u, u' in Left, v, v' in Right
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
                
                # Check validity of swap (u, v), (u', v') -> (u, v'), (u', v)
                # u, u' are Left, v, v' are Right
                if v_prime in adj[u]:
                    continue
                if v in adj[u_prime]:
                    continue
                
                # Perform swap
                # Safety check: ensure elements are present
                if v not in adj[u] or v_prime not in adj[u_prime]:
                    continue
                if u not in adj[v] or u_prime not in adj[v_prime]:
                    continue

                adj[u].remove(v)
                adj[u].append(v_prime)
                adj[v].remove(u)
                adj[v].append(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                
                new_girth = get_girth(adj)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth >= 10:
                        edges = []
                        for u_node in range(n // 2):
                            for v_node in adj[u_node]:
                                if u_node < v_node:
                                    edges.append((u_node, v_node))
                        return n, edges
                else:
                    # Revert
                    # Safety check: ensure elements are present
                    if v_prime not in adj[u] or v not in adj[u_prime]:
                        continue
                    if u_prime not in adj[v] or u not in adj[v_prime]:
                        continue

                    adj[u].remove(v_prime)
                    adj[u].append(v)
                    adj[v].remove(u_prime)
                    adj[v].append(u)
                    
                    adj[u_prime].remove(v)
                    adj[u_prime].append(v_prime)
                    adj[v_prime].remove(u)
                    adj[v_prime].append(u_prime)

            return None

        # Try specific n values
        # 320 is the cage.
        # If we fail, try larger n where it's easier to find a high girth graph?
        # Actually, for fixed k, random graphs have girth ~ log_k n.
        # For k=4, log_4(320) ~ 4.5. So random 320-vertex graph has girth ~ 8-10.
        # It's close.
        
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400, 500]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Allocate time
            # Give 320 more time
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
    # Ensure this part is robust
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
