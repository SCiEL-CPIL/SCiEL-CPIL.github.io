import random
from collections import deque
import time

def construct(k, g):
    # For k=4, g=10, use voltage assignment on K_4,4
    if k != 4 or g != 10:
        # Fallback for other (k, g)
        return _fallback_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin
    
    # Base Graph: K_4,4
    # Vertices 0-3 (left), 4-7 (right)
    # Edges: (i, j+4) for i in 0..3, j in 0..3
    base_edges = []
    base_adj = [[] for _ in range(8)]
    for i in range(4):
        for j in range(4):
            u = i
            v = j + 4
            base_edges.append((u, v))
            base_adj[u].append(v)
            base_adj[v].append(u)
    
    # Find all cycles of length < 10 in K_4,4
    # K_4,4 is bipartite, so only even cycles.
    # Lengths: 4, 6, 8.
    short_cycles = []
    
    # Find 4-cycles
    # A 4-cycle is u-v-w-x-u where u,x in Left, v,w in Right
    # u != x, v != w
    for u1 in range(4):
        for u2 in range(u1 + 1, 4):
            for v1 in range(4):
                for v2 in range(v1 + 1, 4):
                    # Cycle: u1 - (v1+4) - u2 - (v2+4) - u1
                    short_cycles.append((u1, v1 + 4, u2, v2 + 4))
                    
    # Find 6-cycles
    # u1 - v1 - u2 - v2 - u3 - v3 - u1
    # u1, u2, u3 distinct in Left
    # v1, v2, v3 distinct in Right
    import itertools
    for u_perm in itertools.permutations(range(4), 3):
        for v_perm in itertools.permutations(range(4), 3):
            # Check if this forms a simple cycle
            # u1-v1-u2-v2-u3-v3-u1
            # Edges: (u1, v1+4), (u2, v1+4), (u2, v2+4), (u3, v2+4), (u3, v3+4), (u1, v3+4)
            # All these edges must exist in K_4,4 (they do)
            # We need to ensure it's a simple cycle (vertices distinct)
            # u1, u2, u3 distinct. v1, v2, v3 distinct.
            # This is always a simple 6-cycle in K_4,4.
            short_cycles.append((u_perm[0], v_perm[0] + 4, u_perm[1], v_perm[1] + 4, u_perm[2], v_perm[2] + 4))
            
    # Find 8-cycles
    # u1-v1-u2-v2-u3-v3-u4-v4-u1
    # u1..u4 is a permutation of 0..3
    # v1..v4 is a permutation of 0..3
    for u_perm in itertools.permutations(range(4)):
        for v_perm in itertools.permutations(range(4)):
            # Check if it's a simple cycle
            # In K_4,4, any alternating sequence of distinct Left and Right vertices forms a simple cycle.
            short_cycles.append(tuple(u_perm[i] if i % 2 == 0 else v_perm[i//2] + 4 for i in range(8)))

    # Precompute edge indices for fast access
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Precompute edge lists for each cycle
    cycle_edge_lists = []
    for cycle in short_cycles:
        edges_in_cycle = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges_in_cycle.append(edge_index[(u, v)])
        cycle_edge_lists.append(edges_in_cycle)

    # Search for voltages
    # Try L from 32 downwards? Or upwards?
    # Smaller L means smaller N.
    # But harder to satisfy constraints?
    # Let's try L = 32, 33, ... up to 40.
    # If we find one for L=32, N=256.
    
    for L in range(32, 41):
        if time.time() > deadline:
            break
            
        N = 8 * L
        if N >= 320:
            continue
            
        # Try to find voltage assignment
        found = False
        for attempt in range(2000):
            if time.time() > deadline:
                break
                
            # Initialize voltages randomly
            voltages = [random.randint(0, L - 1) for _ in base_edges]
            
            # Local Search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Check all constraints
                for c_idx, c_edges in enumerate(cycle_edge_lists):
                    s = sum(voltages[e] for e in c_edges) % L
                    if s == 0:
                        # Violation found
                        # Try to fix by changing one voltage in the cycle
                        # Pick a random edge in the cycle and change its voltage
                        e_idx = random.choice(c_edges)
                        old_v = voltages[e_idx]
                        # Try a new value
                        new_v = (old_v + random.randint(1, L - 1)) % L
                        if new_v != old_v:
                            voltages[e_idx] = new_v
                            improved = True
                            # Check if this created new violations? 
                            # Simple hill climbing might loop. 
                            # Better: Check if total violations decreased.
                            # For now, just accept if it fixes this one.
                            # To avoid oscillation, we might need a smarter move.
                            pass
        
        # After local search, check if valid
        valid = True
        for c_edges in cycle_edge_lists:
            s = sum(voltages[e] for e in c_edges) % L
            if s == 0:
                valid = False
                break
                
        if valid:
            # Build graph
            adj = [[] for _ in range(N)]
            for i, (u, v) in enumerate(base_edges):
                vol = voltages[i]
                for offset in range(L):
                    u_lift = u * L + offset
                    v_lift = v * L + (offset + vol) % L
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            if _check_girth(adj, N, g):
                edges = []
                for u in range(N):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return N, edges
                
    # Fallback: LUW graph
    return _fallback_construct(k, g)


def _check_girth(adj, n, girth):
    """Check if graph has no cycle shorter than girth using non-backtracking BFS."""
    for start in range(n):
        # State: (current, prev) -> depth
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb))
            visited[(start, nb)] = 1
        
        while queue:
            cur, prev = queue.popleft()
            depth = visited[(cur, prev)]
            if depth + 1 >= girth:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                new_state = (nb, cur)
                if new_state in visited:
                    d2 = visited[new_state]
                    if depth + 1 + d2 < girth:
                        return False
                else:
                    visited[new_state] = depth + 1
                    if depth + 1 < girth:
                        queue.append(new_state)
    return True


def _fallback_construct(k, g):
    """Fallback to LUW-style construction."""
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
