import random
import time
from collections import defaultdict

_T0 = time.time()
_DEADLINE = _T0 + 145.0

def construct(k, g):
    if k == 3 and g == 16:
        return _construct_petersen_lift()
    else:
        return _luw_construct(k, g)

def _construct_petersen_lift():
    # Petersen graph definition
    # Vertices 0-9
    # Edges:
    # Outer: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    
    adj = [[] for _ in range(10)]
    edges = []
    def add_edge(u, v):
        idx = len(edges)
        edges.append((u, v))
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Outer
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(2, 3)
    add_edge(3, 4)
    add_edge(4, 0)
    # Inner
    add_edge(5, 7)
    add_edge(7, 9)
    add_edge(9, 6)
    add_edge(6, 8)
    add_edge(8, 5)
    # Spokes
    add_edge(0, 5)
    add_edge(1, 6)
    add_edge(2, 7)
    add_edge(3, 8)
    add_edge(4, 9)
    
    num_edges = len(edges)
    
    # Generate non-backtracking closed walks of length 5 to 15
    # Walk is a sequence of edge indices.
    # Start at vertex 0, edge e.
    # To avoid duplicates, we can just generate all and use a set of tuples?
    # Or just generate and filter.
    
    walks = []
    # We need to track (current_vertex, previous_edge)
    # Start from each vertex and each incident edge?
    # Actually, a closed walk is determined by its sequence of edges.
    # We can start DFS from (0, e) for each edge e incident to 0?
    # No, that only finds walks starting at 0.
    # We need all closed walks.
    # A closed walk is a cycle in the line graph?
    # Let's just generate all non-backtracking walks of length L starting from every (v, e) and check if they end at v.
    
    # Optimization:
    # Precompute transitions:
    # trans[v][e_in] = list of (e_out, v_next) such that e_in != e_out and e_in, e_out incident to v.
    
    trans = defaultdict(list)
    for v in range(10):
        for nb, e in adj[v]:
            for nb2, e2 in adj[v]:
                if e != e2:
                    trans[(v, e)].append((e2, nb2))
    
    for L in range(5, 16):
        # Start from all (v, e)
        for v in range(10):
            for nb, e in adj[v]:
                # State: (current_vertex, last_edge)
                # We want to close the walk, so we need to track the start?
                # A walk is closed if it returns to the start vertex.
                # But "non-backtracking" implies we don't immediately go back.
                # We can represent a walk by its edge sequence.
                # Let's do DFS on (v, last_e, length, path_edges)
                
                # To avoid massive memory, we can use a recursive generator or iterative stack.
                # Since L is small (<=15), stack is fine.
                
                stack = [(v, e, 0, [e])]
                while stack:
                    cv, le, depth, path = stack.pop()
                    if depth == L:
                        # Check if closed: current vertex must be start vertex?
                        # Wait, the path started at some vertex.
                        # We need to know the start vertex.
                        # Let's restart: Start from (start_v, start_e).
                        pass
                
                # Better approach:
                # Iterate all possible starting points (s_v, s_e).
                # DFS for length L.
                # If we end at s_v, it's a closed walk.
                # To avoid duplicates (same cycle different start), we can just collect all and deduplicate?
                # Or just collect all. The number of walks is manageable.
                
                start_v = v
                start_e = e
                stack = [(start_v, start_e, 1, [start_e])]
                while stack:
                    cv, le, depth, path = stack.pop()
                    if depth == L:
                        if cv == start_v:
                            # Found a closed walk
                            # Normalize: rotate to smallest edge index?
                            # Or just add.
                            # We need to check if this walk is already in `walks`?
                            # For verification, we just need a set of constraints.
                            # Duplicates are fine, just redundant checks.
                            walks.append(tuple(path))
                    else:
                        for ne, nv in trans[(cv, le)]:
                            stack.append((nv, ne, depth + 1, path + [ne]))
    
    # Deduplicate walks to save time?
    # There might be many duplicates.
    # Let's use a set.
    unique_walks = list(set(walks))
    
    # Map edge to list of walk indices
    edge_to_walks = defaultdict(list)
    for i, w in enumerate(unique_walks):
        for e in w:
            edge_to_walks[e].append(i)
            
    num_walks = len(unique_walks)
    
    best_n = None
    best_edges = None
    
    # Try moduli m
    # Lower bound n >= 512 => 10m >= 512 => m >= 52
    # Upper bound: try up to 100 (n=1000)
    
    for m in range(52, 101):
        if time.time() > _DEADLINE - 5:
            break
            
        # Random initial voltages
        voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
        
        # Compute initial sums
        sums = [0] * num_walks
        for i, w in enumerate(unique_walks):
            s = 0
            for e in w:
                s += voltages[e]
            sums[i] = s % m
            
        # Count bad walks
        bad_count = sum(1 for s in sums if s == 0)
        
        # Local Search
        # We want to reduce bad_count to 0.
        # If bad_count == 0, success.
        
        if bad_count == 0:
            # Found valid graph
            best_n = 10 * m
            # Construct edges
            lift_edges = []
            # Vertex in lift: (v, h) where h is the voltage offset.
            # Edge (u, v) with voltage val connects (u, h) to (v, h + val)
            # We need to list all edges.
            for u, v in edges:
                # We need the voltage of this edge.
                # We stored voltages by edge index.
                # Let's get index.
                # We need a map from (u,v) to index.
                pass
            
            # Re-constructing edges from voltages:
            # We have `edges` list of (u,v).
            # We have `voltages` list.
            # For each base edge (u, v) with index i and voltage val:
            # It connects (u, h) to (v, h + val) for all h in 0..m-1.
            # Total edges: 15 * m.
            # Vertices: 10 * m.
            
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                val = voltages[i]
                for h in range(m):
                    u_lift = u * m + h
                    v_lift = v * m + (h + val) % m
                    lift_edges.append((u_lift, v_lift))
            
            best_n = 10 * m
            best_edges = lift_edges
            
            # Check if we can find smaller m
            # Since we iterate m from 52 up, the first one found is the smallest?
            # Yes. So we can return immediately?
            # But maybe 52 fails, 53 fails... 94 succeeds.
            # We want the smallest n. So we should return the first success.
            
            # However, if we find one at 94, but there is one at 60 that we missed due to bad luck?
            # We should probably run the search for a few seconds for each m.
            # If we find a valid one, we return it.
            # If we don't, we try next m.
            
            # To be safe, let's run the search for a limited time per m.
            # If we find a valid one, we return.
            
            return best_n, best_edges
            
        # If not found immediately, run local search
        # Time budget per m: 5 seconds?
        start_time = time.time()
        while time.time() - start_time < 5.0 and bad_count > 0:
            if time.time() > _DEADLINE - 2:
                break
                
            # Pick a random bad walk
            # Or pick a random edge involved in a bad walk?
            # Let's pick a random bad walk.
            bad_indices = [i for i, s in enumerate(sums) if s == 0]
            if not bad_indices:
                break
            w_idx = random.choice(bad_indices)
            w = unique_walks[w_idx]
            
            # Pick a random edge in this walk to change
            e_idx = random.choice(w)
            
            # Change voltage of e_idx
            # Try a small change? Or random?
            # Random change in Z_m
            old_val = voltages[e_idx]
            new_val = random.randint(0, m - 1)
            if new_val == old_val:
                continue
                
            delta = (new_val - old_val) % m
            voltages[e_idx] = new_val
            
            # Update sums for all walks containing e_idx
            for w_i in edge_to_walks[e_idx]:
                sums[w_i] = (sums[w_i] + delta) % m
                # Check if it became 0 or non-0
                # We need to update bad_count
                # This is expensive if we recompute bad_count.
                # Let's just recompute bad_count periodically or use a set of bad indices.
                
            # Recompute bad_count?
            # It's O(num_walks). num_walks ~ 10^5.
            # 10^5 operations per step.
            # 5 seconds / (10^5 * 10ns) = 5000 steps?
            # Should be enough.
            
            bad_count = sum(1 for s in sums if s == 0)
            
            if bad_count == 0:
                # Found valid graph
                lift_edges = []
                for i, (u, v) in enumerate(edges):
                    val = voltages[i]
                    for h in range(m):
                        u_lift = u * m + h
                        v_lift = v * m + (h + val) % m
                        lift_edges.append((u_lift, v_lift))
                return 10 * m, lift_edges
                
    # If no m worked, fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
