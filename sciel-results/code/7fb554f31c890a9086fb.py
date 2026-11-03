import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, uses vectorized girth checking with simulated annealing
    and targeted cycle-breaking to find a 320-vertex cage.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    if k != 4 or g != 10:
        return _luw_construct(k, g)

    target_n = 320

    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices."""
        for attempt in range(100):
            stubs = list(range(n)) * 4
            random.shuffle(stubs)
            pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
            
            # Fix loops
            valid = True
            for i in range(len(pairs)):
                if pairs[i][0] == pairs[i][1]:
                    u = pairs[i][0]
                    fixed = False
                    for j in range(len(pairs)):
                        if j == i:
                            continue
                        a, b = pairs[j]
                        if a != u and b != u:
                            pairs[i] = (u, a)
                            pairs[j] = (u, b)
                            fixed = True
                            break
                    if not fixed:
                        valid = False
                        break
            if not valid:
                continue
            
            # Check multi-edges
            edge_set = set()
            valid = True
            for u, v in pairs:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if valid:
                return pairs
        return None

    def _build_adj_np(n, edges):
        """Build NumPy adjacency matrix and list from edge list."""
        adj_np = np.zeros((n, n), dtype=np.int8)
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_np[u, v] = 1
            adj_np[v, u] = 1
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_np, adj_list

    def _count_short_cycles_fast(n, adj_np, g):
        """
        Vectorized count of non-backtracking walks of length < g returning to start.
        Uses matrix operations for speed.
        """
        # For girth g, we need no cycles of length 3..g-1
        # Count = sum over s of (number of non-backtracking closed walks of length l for l in 3..g-1)
        # Non-backtracking matrix: B = A - D^{-1} (but for regular graphs, simpler)
        # For k-regular: non-backtracking closed walks count via trace of B^l
        # B_{uv} = A_{uv} - 1/(k-1) * sum over w != u,v of A_{uw}*A_{vw}
        # For regular graphs: B = A - (1/(k-1)) * (J - I - A) ... this gets complex
        # Simpler: use the fact that for counting, we can do BFS-like with numpy
        
        total = 0
        k = 4
        # Non-backtracking walk count: W_l(s) = number of non-backtracking walks of length l from s back to s
        # W_1 = 0 (no self-loops), W_2 = k (go out and back)
        # For l >= 3: W_l(s) = k * (k-1)^(l-1) - ... no, that's for trees
        # Actually for counting cycles: a cycle of length l contributes l to the count of closed walks
        # Non-backtracking closed walk of length l from s: count = trace of (B)^l where B is non-backtracking matrix
        
        # Build non-backtracking matrix B
        # B_{u,v} = 1 if A_{u,v}=1, else 0; but we need to exclude backtracking
        # Standard: B = A - D^{-1}(A^2 - A) for simple graphs... 
        # Actually the non-backtracking matrix is: B_{uv} = A_{uv} - (1/(k-1)) * sum_{w!=u,v} A_{uw}A_{vw}
        # For regular graphs this simplifies. Let's just use the direct formula:
        # (B^l)_{ss} counts non-backtracking closed walks of length l from s to s
        # A cycle of length l is counted l times in the trace (once for each starting point)
        # So total cycle count of length l = trace(B^l) / l
        
        # For k=4: B = A - (1/3)(A^2 - A) ... no
        # The non-backtracking matrix for a k-regular graph:
        # B = A - (1/(k-1)) * (A^2 - A) is NOT correct in general
        # Correct: B_{uv} = A_{uv} for u!=v, and we define the walk counting differently
        # Let me use the standard result: for a k-regular graph,
        # the number of non-backtracking closed walks of length l is trace(B^l)
        # where B is the Hashimoto/non-backtracking matrix, but that's 2n x 2n
        # 
        # Simpler approach: just count using the adjacency matrix with the correction
        # For a k-regular graph, the number of closed walks of length l that don't backtrack
        # can be computed recursively. But for speed, let's use:
        # trace(A^l) counts all closed walks. For regular graphs, we can subtract backtracking ones.
        # 
        # Actually, the simplest fast approach: for each length l from 3 to g-1,
        # compute trace of the non-backtracking walk matrix
        # For k-regular: N_l = (1/(k-1)) * (trace(A^l) - k * (k-1)^(l-1) * n) ... not quite
        #
        # Let me just do a fast BFS-based count with numpy for the early termination
        
        # Fast method: for each start vertex, do a breadth-first non-backtracking search
        # and count returns. Use numpy for the frontier.
        
        count = 0
        # We'll do this per-vertex but with numpy vectorization where possible
        # For n=320, g=10, max depth 9, this should be fast enough
        
        for s in range(n):
            # State: (current_node, prev_node)
            # Use a dict for visited (node, prev) pairs
            visited = np.zeros((n, n), dtype=bool)
            visited[s, s] = True  # (s, -1) represented as (s, s)
            
            # Frontier: list of (node, prev)
            frontier = [(s, s)]
            
            for depth in range(1, g):
                if not frontier:
                    break
                next_frontier = []
                for u, prev in frontier:
                    for v in range(n):
                        if adj_np[u, v] == 0:
                            continue
                        if v == prev:
                            continue
                        if v == s and depth < g:
                            # Found a cycle of length depth
                            # But we need depth >= 3 for it to be a real cycle
                            if depth >= 3:
                                count += 1
                        if not visited[v, u]:
                            visited[v, u] = True
                            next_frontier.append((v, u))
                frontier = next_frontier
                if depth >= g - 1:
                    break
        
        return count

    def _count_short_cycles_bfs(n, adj_list, g):
        """BFS-based count of short cycles using non-backtracking walks."""
        count = 0
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj_list[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g and d + 1 >= 3:
                        count += 1
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, d + 1))
        return count

    def _find_short_cycle(n, adj_list, g):
        """Find one short cycle if it exists. Returns list of vertices or None."""
        for s in range(n):
            visited = {}
            visited[(s, -1)] = None
            queue = deque([(s, -1, 0, [s])])
            while queue:
                u, prev, d, path = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj_list[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g and d + 1 >= 3:
                        return path + [s]
                    if (v, u) not in visited:
                        visited[(v, u)] = (u, prev)
                        queue.append((v, u, d + 1, path + [v]))
        return None

    def _break_cycle_swap(n, adj_list, edge_set, edges, cycle):
        """
        Try to break a short cycle by swapping edges.
        Returns True if a valid swap was made.
        """
        if len(cycle) < 4:
            return False
        
        # The cycle is v0-v1-v2-...-v_{l-1}-v0
        # Try to swap two non-adjacent edges on the cycle
        l = len(cycle) - 1  # cycle length
        if l < 4:
            return False
        
        # Pick two edges on the cycle that are not adjacent
        # Edge i is (cycle[i], cycle[i+1]) for i in 0..l-1 (with cycle[l]=cycle[0])
        for i in range(l):
            for j in range(i + 2, l):
                if i == 0 and j == l - 1:
                    continue  # adjacent edges
                # Edge i: (cycle[i], cycle[i+1])
                # Edge j: (cycle[j], cycle[j+1]) where cycle[l] = cycle[0]
                u1, v1 = cycle[i], cycle[(i+1) % l]
                u2, v2 = cycle[j], cycle[(j+1) % l]
                
                # Try swap: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
                new_e1 = (u1, u2)
                new_e2 = (v1, v2)
                
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                
                ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                
                if ne1 == ne2:
                    continue
                
                old_e1 = (min(u1, v1), max(u1, v1))
                old_e2 = (min(u2, v2), max(u2, v2))
                
                if old_e1 in edge_set and old_e2 in edge_set:
                    if ne1 not in edge_set and ne2 not in edge_set:
                        # Apply swap
                        adj_list[u1].remove(v1)
                        adj_list[v1].remove(u1)
                        adj_list[u2].remove(v2)
                        adj_list[v2].remove(u2)
                        edge_set.discard(old_e1)
                        edge_set.discard(old_e2)
                        
                        adj_list[u1].append(u2)
                        adj_list[u2].append(u1)
                        adj_list[v1].append(v2)
                        adj_list[v2].append(v1)
                        edge_set.add(ne1)
                        edge_set.add(ne2)
                        
                        # Update edges list
                        for idx, (a, b) in enumerate(edges):
                            if (min(a,b), max(a,b)) == old_e1:
                                edges[idx] = ne1
                                break
                        for idx, (a, b) in enumerate(edges):
                            if (min(a,b), max(a,b)) == old_e2:
                                edges[idx] = ne2
                                break
                        return True
        return False

    edges = _gen_random_4reg(target_n)
    if edges is None:
        return _luw_construct(k, g)

    adj_np, adj_list = _build_adj_np(target_n, edges)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    current_energy = _count_short_cycles_bfs(target_n, adj_list, g)
    
    T = 2.0
    cooling = 0.999
    min_T = 0.05
    best_energy = current_energy
    best_edges = edges[:]
    
    # Phase 1: Targeted cycle breaking
    phase1_deadline = start_time + 30
    while time.time() < phase1_deadline and current_energy > 0:
        cycle = _find_short_cycle(target_n, adj_list, g)
        if cycle is None:
            current_energy = 0
            break
        
        if _break_cycle_swap(target_n, adj_list, edge_set, edges, cycle):
            new_energy = _count_short_cycles_bfs(target_n, adj_list, g)
            if new_energy < best_energy:
                best_energy = new_energy
                best_edges = edges[:]
            if new_energy < current_energy:
                current_energy = new_energy
            if current_energy == 0:
                break
        else:
            # If can't break this cycle, try random swaps
            for trial in range(10):
                i1 = random.randint(0, len(edges) - 1)
                i2 = random.randint(0, len(edges) - 1)
                if i1 == i2:
                    continue
                u1, v1 = edges[i1]
                u2, v2 = edges[i2]
                new_e1 = (u1, u2)
                new_e2 = (v1, v2)
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                if ne1 == ne2:
                    continue
                old_e1 = (min(u1, v1), max(u1, v1))
                old_e2 = (min(u2, v2), max(u2, v2))
                if old_e1 in edge_set and old_e2 in edge_set:
                    if ne1 not in edge_set and ne2 not in edge_set:
                        # Apply
                        adj_list[u1].remove(v1)
                        adj_list[v1].remove(u1)
                        adj_list[u2].remove(v2)
                        adj_list[v2].remove(u2)
                        edge_set.discard(old_e1)
                        edge_set.discard(old_e2)
                        adj_list[u1].append(u2)
                        adj_list[u2].append(u1)
                        adj_list[v1].append(v2)
                        adj_list[v2].append(v1)
                        edge_set.add(ne1)
                        edge_set.add(ne2)
                        edges[i1] = ne1
                        edges[i2] = ne2
                        
                        new_energy = _count_short_cycles_bfs(target_n, adj_list, g)
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_edges = edges[:]
                        if new_energy < current_energy or random.random() < math.exp(-(new_energy - current_energy) / T):
                            current_energy = new_energy
                        else:
                            # Revert
                            adj_list[u1].remove(u2)
                            adj_list[u2].remove(u1)
                            adj_list[v1].remove(v2)
                            adj_list[v2].remove(v1)
                            edge_set.discard(ne1)
                            edge_set.discard(ne2)
                            adj_list[u1].append(v1)
                            adj_list[v1].append(u1)
                            adj_list[u2].append(v2)
                            adj_list[v2].append(u2)
                            edge_set.add(old_e1)
                            edge_set.add(old_e2)
                            edges[i1] = old_e1
                            edges[i2] = old_e2
                        break
        
        T *= cooling
        if T < min_T:
            T = min_T

    if current_energy == 0:
        return target_n, edges

    # Phase 2: Simulated annealing with random swaps
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        i1 = random.randint(0, len(edges) - 1)
        i2 = random.randint(0, len(edges) - 1)
        if i1 == i2:
            continue
        
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]
        
        new_e1 = (u1, u2)
        new_e2 = (v1, v2)
        
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        
        ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
        ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
        
        if ne1 == ne2:
            continue
        
        old_e1 = (min(u1, v1), max(u1, v1))
        old_e2 = (min(u2, v2), max(u2, v2))
        
        if old_e1 in edge_set and old_e2 in edge_set:
            if ne1 not in edge_set and ne2 not in edge_set:
                # Apply
                adj_list[u1].remove(v1)
                adj_list[v1].remove(u1)
                adj_list[u2].remove(v2)
                adj_list[v2].remove(u2)
                edge_set.discard(old_e1)
                edge_set.discard(old_e2)
                adj_list[u1].append(u2)
                adj_list[u2].append(u1)
                adj_list[v1].append(v2)
                adj_list[v2].append(v1)
                edge_set.add(ne1)
                edge_set.add(ne2)
                edges[i1] = ne1
                edges[i2] = ne2
                
                new_energy = _count_short_cycles_bfs(target_n, adj_list, g)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_edges = edges[:]
                    if new_energy == 0:
                        return target_n, edges
                
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current_energy = new_energy
                else:
                    # Revert
                    adj_list[u1].remove(u2)
                    adj_list[u2].remove(u1)
                    adj_list[v1].remove(v2)
                    adj_list[v2].remove(v1)
                    edge_set.discard(ne1)
                    edge_set.discard(ne2)
                    adj_list[u1].append(v1)
                    adj_list[v1].append(u1)
                    adj_list[u2].append(v2)
                    adj_list[v2].append(u2)
                    edge_set.add(old_e1)
                    edge_set.add(old_e2)
                    edges[i1] = old_e1
                    edges[i2] = old_e2
                
                T *= cooling

    if best_energy == 0:
        return target_n, best_edges

    # Final check: if we have a good but not perfect result, try a few more targeted breaks
    if time.time() < deadline - 5:
        adj_list_final = [[] for _ in range(target_n)]
        for u, v in best_edges:
            adj_list_final[u].append(v)
            adj_list_final[v].append(u)
        edge_set_final = set((min(u, v), max(u, v)) for u, v in best_edges)
        
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            cycle = _find_short_cycle(target_n, adj_list_final, g)
            if cycle is None:
                return target_n, best_edges
            if _break_cycle_swap(target_n, adj_list_final, edge_set_final, best_edges, cycle):
                # Rebuild adj from edges
                adj_list_final = [[] for _ in range(target_n)]
                for u, v in best_edges:
                    adj_list_final[u].append(v)
                    adj_list_final[v].append(u)
                edge_set_final = set((min(u, v), max(u, v)) for u, v in best_edges)

    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
