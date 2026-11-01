import time
from collections import deque

def construct(k, g):
    # Petersen graph lift with Thue-Morse-derived voltage assignments
    # to structurally minimize low-order cycle collisions in Z_m
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Precompute Thue-Morse sequence (long enough for our needs)
    def generate_thue_morse(n):
        tm = [0] * n
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    tm_seq = generate_thue_morse(200)
    
    start_time = time.time()
    target_girth = g  # typically 16
    
    # Try increasing m values
    for m in range(8, 200):
        if time.time() - start_time > 100:
            break
        
        # Generate multiple Thue-Morse-derived voltage patterns
        # Key insight: use different "phases" and "scales" of the Thue-Morse word
        # to create high-dispersion assignments that avoid small cycle sums
        
        voltage_patterns = []
        
        # Pattern A: Direct Thue-Morse bits scaled
        for shift in range(0, 15, 3):
            pattern = [(tm_seq[(i + shift) % 100] * (i + 1)) % m for i in range(15)]
            voltage_patterns.append(pattern)
        
        # Pattern B: Thue-Morse with multiplicative spread
        for scale in [1, 2, 3, 5, 7, 11]:
            pattern = [(tm_seq[i] * scale * (i + 1)) % m for i in range(15)]
            voltage_patterns.append(pattern)
        
        # Pattern C: Quadratic Thue-Morse mixing
        for c in [1, 2, 3]:
            pattern = [(tm_seq[i] * (i + c) * (i + c + 1)) % m for i in range(15)]
            voltage_patterns.append(pattern)
        
        # Pattern D: Alternating Thue-Morse blocks
        for block_size in [3, 5, 7]:
            pattern = []
            for i in range(15):
                block = i // block_size
                pos = i % block_size
                val = tm_seq[(block * 13 + pos) % 100]
                pattern.append((val * (block + 1) * (pos + 1)) % m)
            voltage_patterns.append(pattern)
        
        # Deduplicate patterns
        seen = set()
        unique_patterns = []
        for p in voltage_patterns:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        
        for vol in unique_patterns:
            if time.time() - start_time > 105:
                break
            
            # Quick girth check: verify no cycle shorter than target_girth
            if _check_girth(petersen_edges, vol, m, target_girth):
                # Build the lifted graph
                edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    w = vol[i] % m
                    for s in range(m):
                        u_id = u * m + s
                        v_id = v * m + (s + w) % m
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                        else:
                            edges.append((v_id, u_id))
                return 10 * m, edges
    
    # Fallback: L-U-W construction
    return _luw_construct(k, g)

def _check_girth(base_edges, voltages, m, target_girth):
    """Check if the voltage lift has girth at least target_girth."""
    n_vertices = 10 * m
    
    # Build adjacency list
    adj = [[] for _ in range(n_vertices)]
    for i, (u, v) in enumerate(base_edges):
        w = voltages[i] % m
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            adj[u_id].append(v_id)
            adj[v_id].append(u_id)
    
    # BFS from each vertex to detect cycles shorter than target_girth
    # Optimization: only need to check up to depth target_girth - 1
    for start in range(n_vertices):
        if time.time() - _check_girth._start > 95:
            return False  # timeout safety
        if not _bfs_check_cycle(start, adj, target_girth):
            return False
    
    return True

_check_girth._start = time.time()

def _bfs_check_cycle(start, adj, target_girth):
    """BFS from start to detect any cycle of length < target_girth."""
    visited = {start: (0, -1)}  # node -> (dist, parent)
    queue = deque([(start, 0, -1)])
    
    while queue:
        node, dist, parent = queue.popleft()
        if dist >= target_girth - 1:
            continue
        
        for neighbor in adj[node]:
            if neighbor == parent:
                continue
            if neighbor in visited:
                vdist, vparent = visited[neighbor]
                # Found a cycle: node -> ... -> neighbor -> ... -> node
                # Cycle length is dist + 1 + vdist if the paths are disjoint
                # But we need to be careful: this detects SOME cycle, not necessarily the shortest
                # For our purposes, if we find any back-edge within depth target_girth-1,
                # there exists a cycle of length <= dist + 1 + vdist
                cycle_len = dist + 1 + vdist
                if cycle_len < target_girth:
                    return False
            else:
                visited[neighbor] = (dist + 1, node)
                queue.append((neighbor, dist + 1, node))
    
    return True

def _luw_construct(k, g):
    """L-U-W fallback construction."""
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
