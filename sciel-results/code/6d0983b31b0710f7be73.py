import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Fallback: LUW graph construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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

    # Petersen graph base
    def get_petersen():
        # Petersen: outer 5-cycle, inner star, spokes
        # Vertices 0-4: outer, 5-9: inner
        # Edges: outer cycle, inner star, spokes
        edges = []
        # Outer cycle: 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner star: 5-7-9-6-8-5
        inner = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner[i], inner[(i + 2) % 5]))
        # Spokes: i - (i+5)
        for i in range(5):
            edges.append((i, i + 5))
        return edges

    def get_petersen_adj(p_edges):
        adj = [[] for _ in range(10)]
        for idx, (u, v) in enumerate(p_edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        return adj

    # Enumerate closed walks of length L in Petersen
    def enumerate_closed_walks(adj, max_len):
        """Return list of (length, edge_indices_list) for all closed walks of length < max_len."""
        walks = []
        for start in range(10):
            # DFS: (current_vertex, path_of_edge_indices)
            stack = [(start, [])]
            while stack:
                v, path = stack.pop()
                L = len(path)
                if L > 0 and v == start and L < max_len:
                    walks.append((L, list(path)))
                if L < max_len:
                    for w, eidx in adj[v]:
                        if not path or w != path[-1]:  # non-backtracking
                            stack.append((w, path + [eidx]))
        return walks

    # Try Petersen lift for k=4, g=10
    if k == 4 and g == 10:
        p_edges = get_petersen()
        p_adj = get_petersen_adj(p_edges)
        num_edges = len(p_edges)  # 15
        
        # Enumerate all closed walks of length 5..9
        closed_walks = enumerate_closed_walks(p_adj, 10)
        # Filter to only length >= 5 (Petersen girth is 5)
        closed_walks = [(L, eidxs) for L, eidxs in closed_walks if L >= 5]
        
        # Precompute edge index sequences for each walk
        walk_seqs = [eidxs for L, eidxs in closed_walks]
        walk_lens = [L for L, eidxs in closed_walks]
        
        # For a given m and voltages v[0..14], check if any walk sums to 0 mod m
        def check_valid(m, voltages):
            for seq in walk_seqs:
                s = 0
                for e in seq:
                    s += voltages[e]
                if s % m == 0:
                    return False
            return True

        def count_bad(m, voltages):
            count = 0
            for seq in walk_seqs:
                s = 0
                for e in seq:
                    s += voltages[e]
                if s % m == 0:
                    count += 1
            return count

        # Search for valid (m, voltages)
        # Try various m values
        random.seed(12345)
        best_result = None
        best_n = float('inf')
        
        # Candidate m values (want to minimize 10*m)
        # m must be large enough that random sums avoid 0 mod m
        # With ~800 walks, for random voltages, P(specific walk sums to 0) ≈ 1/m
        # Expected bad walks ≈ 800/m. Need this < 1 for likely success.
        # So m > 800 is safe, but smaller m might work with good voltages.
        # Let's try m from small to large, stop when we find valid.
        
        m_candidates = list(range(100, 200)) + list(range(200, 400, 10)) + list(range(400, 800, 20)) + list(range(800, 1500, 50))
        
        for m in m_candidates:
            if time.time() > deadline:
                break
            n = 10 * m
            if n >= best_n:
                break
            
            # Try random voltage assignments
            found = False
            num_trials = 500
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                voltages = [random.randint(1, m - 1) for _ in range(num_edges)]
                if check_valid(m, voltages):
                    best_result = (m, voltages)
                    best_n = n
                    found = True
                    break
            
            if found:
                # Try to find smaller m
                continue
        
        # If not found with random, try local search for best m
        if best_result is None:
            for m in [100, 150, 200, 250, 300]:
                if time.time() > deadline:
                    break
                n = 10 * m
                if n >= best_n:
                    break
                
                # Local search
                voltages = [random.randint(1, m - 1) for _ in range(num_edges)]
                best_count = count_bad(m, voltages)
                
                for iteration in range(2000):
                    if time.time() > deadline:
                        break
                    if best_count == 0:
                        best_result = (m, voltages)
                        best_n = n
                        break
                    
                    # Try changing one voltage
                    improved = False
                    for eidx in range(num_edges):
                        if time.time() > deadline:
                            break
                        old_val = voltages[eidx]
                        for new_val in range(1, m):
                            if new_val == old_val:
                                continue
                            voltages[eidx] = new_val
                            cnt = count_bad(m, voltages)
                            if cnt < best_count:
                                best_count = cnt
                                improved = True
                            else:
                                voltages[eidx] = old_val
                                break
                        if improved:
                            break
                        voltages[eidx] = old_val
                    
                    if not improved:
                        # Restart
                        voltages = [random.randint(1, m - 1) for _ in range(num_edges)]
                        best_count = count_bad(m, voltages)
                
                if best_count == 0:
                    best_result = (m, voltages)
                    best_n = n
        
        if best_result is not None:
            m, voltages = best_result
            n = 10 * m
            edges = []
            for idx, (u, v) in enumerate(p_edges):
                w = voltages[idx]
                for g_val in range(m):
                    src = u * m + g_val
                    dst = v * m + ((g_val + w) % m)
                    if src < dst:
                        edges.append((src, dst))
            return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
