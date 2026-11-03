import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 118.0  # 2s margin before 120s kill
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
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
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check_fast(n, adj, g):
        """Fast non-backtracking walk girth check from every vertex using arrays."""
        # For each start vertex s, do non-backtracking BFS
        # State: (node, prev_node), distance
        # Use a 2D array: dist[node][prev_node] = distance, or -1 if unvisited
        # prev_node is in 0..n-1, plus a special "no prev" state (use n as sentinel)
        
        for s in range(n):
            # dist[u][p] = shortest non-backtracking path length from (s,-1) to (u,p)
            # p in 0..n-1, where p is the previous node
            # We also need the initial state (s, -1) which has no previous node
            # Use dist[u][p] with p in 0..n-1, and handle (s,-1) separately
            
            # Initialize
            dist = [[-1] * n for _ in range(n)]
            # Initial state: at s, came from "nowhere"
            # We'll represent this as a special case
            
            queue = deque()
            # Start: at vertex s, no previous vertex
            queue.append((s, -1, 0))
            
            # Mark visited states
            visited = set()
            visited.add((s, -1))
            
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    # Check if we found a cycle back to s
                    if v == s and d + 1 < g:
                        return False
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, d + 1))
        return True

    def count_short_cycles_fast(n, adj, g):
        """Count number of short cycles using non-backtracking walks."""
        count = 0
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
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
                        queue.append((v, u, d + 1))
        return count

    def generate_simple_regular_fast(n, k, max_attempts=20):
        """Generate a simple k-regular graph using configuration model."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Create stubs
            stubs = list(range(n)) * k
            random.shuffle(stubs)
            
            # Pair stubs
            edges = []
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u != v:
                    edges.append((u, v))
            
            # Check if already simple
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
            
            # Try repair with 2-switches
            edge_dict = {}
            for i, (u, v) in enumerate(edges):
                key = (min(u, v), max(u, v))
                if key not in edge_dict:
                    edge_dict[key] = []
                edge_dict[key].append(i)
            
            max_swaps = min(n * k // 4, 500)
            for swap_iter in range(max_swaps):
                if time.time() > deadline - 5:
                    return None
                
                # Find a duplicate edge
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
                
                # Try to swap with another edge
                found_swap = False
                for j in random.sample(range(len(edges)), min(30, len(edges))):
                    if j == bad_idx1:
                        continue
                    u2, v2 = edges[j]
                    
                    # Try swapping v1 with v2
                    new_u1, new_v1 = u1, v2
                    new_u2, new_v2 = u2, v1
                    
                    if new_u1 == new_v1 or new_u2 == new_v2:
                        continue
                    
                    key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
                    key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
                    
                    if key1 == bad_key and key2 == bad_key:
                        continue
                    
                    count1 = len(edge_dict.get(key1, []))
                    count2 = len(edge_dict.get(key2, []))
                    
                    old_key1 = (min(u1, v1), max(u1, v1))
                    old_key2 = (min(u2, v2), max(u2, v2))
                    old_count1 = len(edge_dict.get(old_key1, []))
                    old_count2 = len(edge_dict.get(old_key2, []))
                    
                    new_count1 = count1 - (1 if key1 == old_key1 else 0)
                    new_count2 = count2 - (1 if key2 == old_key2 else 0)
                    
                    if new_count1 < 0 or new_count2 < 0:
                        continue
                    
                    # Apply swap
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
            
            # Final check
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

    def local_search_reduce_cycles_fast(n, adj, edges, g, max_iters=300):
        """Local search to reduce number of short cycles with faster evaluation."""
        best_edges = list(edges)
        best_count = count_short_cycles_fast(n, adj, g)
        
        if best_count == 0:
            return best_edges, 0, adj
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if best_count == 0:
                break
            
            # Pick a random edge
            i = random.randint(0, len(best_edges) - 1)
            u1, v1 = best_edges[i]
            
            # Try swapping with a random other edge
            j = random.randint(0, len(best_edges) - 1)
            if j == i:
                continue
            u2, v2 = best_edges[j]
            
            # Try swap: (u1,v1) and (u2,v2) -> (u1,v2) and (u2,v1)
            new_u1, new_v1 = u1, v2
            new_u2, new_v2 = u2, v1
            
            if new_u1 == new_v1 or new_u2 == new_v2:
                continue
            
            # Check for duplicate edges using a set
            key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
            key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
            
            # Quick check: build edge set excluding the two edges we're modifying
            edge_set = set()
            for idx, (u, v) in enumerate(best_edges):
                if idx != i and idx != j:
                    edge_set.add((min(u, v), max(u, v)))
            
            if key1 in edge_set or key2 in edge_set or key1 == key2:
                continue
            
            # Apply swap and check
            best_edges[i] = (new_u1, new_v1)
            best_edges[j] = (new_u2, new_v2)
            
            new_adj = build_adj(best_edges, n)
            new_count = count_short_cycles_fast(n, new_adj, g)
            
            if new_count < best_count:
                best_count = new_count
                adj = new_adj
            else:
                # Revert
                best_edges[i] = (u1, v1)
                best_edges[j] = (u2, v2)
        
        return best_edges, best_count, adj

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # For k=5, g=9: Moore bound is 428, seed is 1116
    # Try to find smaller graphs with girth >= 9
    
    # Strategy: try n values from just above Moore bound up to seed_n
    # Focus on even n (needed for regular graphs)
    search_start = max(moore_bound, 430)
    if search_start % 2 != 0:
        search_start += 1
    search_end = min(seed_n, 700)
    if search_end % 2 != 0:
        search_end -= 1
    
    if search_start <= search_end:
        # Try n values, starting from smaller n
        n_values = list(range(search_start, search_end + 1, 2))
        
        # Limit total number of n values to try based on time
        max_n_values = 25
        
        for n in n_values[:max_n_values]:
            if time.time() > deadline - 15:
                break
            
            # Try up to 2 random graphs for this n
            for attempt in range(2):
                if time.time() > deadline - 15:
                    break
                
                edges = generate_simple_regular_fast(n, k, max_attempts=10)
                if edges is None:
                    continue
                
                adj = build_adj(edges, n)
                
                # Check girth
                if exact_girth_check_fast(n, adj, g):
                    return n, edges
                
                # Local search to reduce cycles
                edges, count, adj = local_search_reduce_cycles_fast(n, adj, edges, g, max_iters=200)
                
                if count == 0:
                    if exact_girth_check_fast(n, adj, g):
                        return n, edges
    
    return seed_n, seed_edges
