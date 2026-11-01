"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # Use a voltage graph (lift) over a small base graph with abelian voltage
    # assignments from a cyclic group Z_m.
    #
    # Base graph: K_6 (complete graph on 6 vertices) is 5-regular.
    # A voltage graph over K_6 with voltages in Z_m gives a 5-regular graph
    # on 6*m vertices.
    #
    # For girth >= 9, we need no non-backtracking closed walk in the lifted
    # graph of length < 9. This is equivalent to: for every non-backtracking
    # closed walk W in the base graph of length L, the sum of voltages along W
    # (with appropriate signs for edge direction) is not 0 mod m.
    #
    # Non-backtracking closed walks in K_6:
    # - Length 3: triangles. There are 20 triangles in K_6.
    # - Length 4: no triangles needed, but walks like a-b-c-a would be backtracking.
    #   Non-backtracking 4-cycles in K_6: a-b-c-d-a where all consecutive pairs are edges
    #   and no immediate reversal.
    # - Length 5: pentagons a-b-c-d-e-a
    # - Length 6: hexagons
    # - Length 7, 8: longer walks
    #
    # The key insight: in K_6, every pair of distinct vertices is connected, so
    # non-backtracking closed walks of length L correspond to sequences of L+1
    # vertices v_0, v_1, ..., v_L = v_0 where v_i != v_{i-1} and v_i != v_{i+2}
    # (no backtracking means v_{i+1} != v_{i-1}).
    #
    # For the voltage to be non-zero mod m for ALL such walks of length < 9,
    # we need to assign voltages carefully.
    #
    # Strategy: Use random voltage assignments and check the constraint.
    # For each candidate m, try random voltage assignments until we find one
    # where no short non-backtracking closed walk has net voltage 0.
    #
    # To make this tractable, we enumerate all non-backtracking closed walks
    # in K_6 up to length 8, and for each, compute the constraint that the
    # voltage sum must be non-zero mod m.
    #
    # Actually, let's think more carefully. The voltage sum for a walk
    # v_0 -> v_1 -> ... -> v_L = v_0 is:
    # sum_{i=0}^{L-1} sign_i * phi(v_i, v_{i+1})
    # where phi(u,v) is the voltage on edge (u,v) (directed u->v), and
    # sign_i = +1 if we traverse in the "forward" direction, -1 if backward.
    #
    # Since K_6 is undirected, we can orient each edge arbitrarily. Let's say
    # for edge (u,v) with u < v, the voltage is phi_uv. If we traverse u->v,
    # we add phi_uv; if v->u, we subtract phi_uv.
    #
    # So for a walk v_0 -> v_1 -> ... -> v_L = v_0:
    # voltage_sum = sum_{i=0}^{L-1} s_i * phi_{v_i, v_{i+1}}
    # where s_i = +1 if v_i < v_{i+1}, else -1, and phi_{v_i,v_{i+1}} is the
    # voltage on the edge with the smaller index first.
    #
    # We need voltage_sum != 0 mod m for all non-backtracking closed walks
    # of length 3, 4, 5, 6, 7, 8.
    #
    # Let's enumerate these walks. In K_6, a non-backtracking closed walk of
    # length L is a cyclic sequence of L vertices (up to rotation and reversal)
    # where no two consecutive vertices are the same and v_i != v_{i+2} mod L.
    #
    # For L=3: triangles. In K_6, these are 3-cycles. Number = C(6,3) = 20.
    #   Each triangle (a,b,c) gives constraint: s_ab*phi_ab + s_bc*phi_bc + s_ca*phi_ca != 0
    #
    # For L=4: 4-cycles with no backtracking. In K_6, a 4-cycle a-b-c-d-a is
    #   non-backtracking iff a != c (which is automatic since a,b,c,d are distinct
    #   in a simple cycle) and b != d. Wait, for a walk of length 4: v0,v1,v2,v3,v4=v0.
    #   Non-backtracking: v1 != v_{-1} = v3, so v1 != v3. Also v2 != v0.
    #   So we need 4 distinct vertices? No: v0,v1,v2,v3 with v0 != v1, v1 != v2,
    #   v2 != v3, v3 != v0 (edges), and v0 != v2, v1 != v3 (no backtracking).
    #   So we need 4 distinct vertices forming a 4-cycle. Number of 4-cycles in K_6:
    #   C(6,4) * 3 = 15 * 3 = 45 (each set of 4 vertices has 3 distinct 4-cycles).
    #
    # This enumeration is getting complex. Let me use a different approach.
    #
    # Alternative: Use the fact that for a voltage graph over a d-regular base,
    # the girth of the lift is at least the girth of the base if the voltages
    # are "sufficiently generic." But K_6 has girth 3, so this doesn't help.
    #
    # Let me try a different base graph. What about a base graph with girth >= 9
    # already? That's the original problem.
    #
    # Actually, the voltage graph approach is meant to construct graphs with
    # specified girth by choosing voltages to avoid short cycles. The base graph
    # can have short cycles, but the voltages "break" them.
    #
    # Let me use a simpler approach: start with the Lazebnik-Ustimenko-Woldar
    # construction from the seed, which gives a valid graph. Then try to use
    # voltage lifts to potentially reduce the number of vertices.
    #
    # Actually, let me reconsider. The seed program uses the L-U-W construction
    # which gives a q-regular bipartite graph with girth at least d+5. For k=5, g=9:
    # q=5, d = max(3, 9-5=4 if even else 9-4=5) = max(3,5) = 5? Wait:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # g=9: g-5=4, 4%2=0, so d = max(3, 9-4) = max(3,5) = 5.
    # The L-U-W graph with d=5, q=5 has girth at least d+5=10.
    # Number of vertices: 2*q^d = 2*5^5 = 6250.
    #
    # That's way larger than 1116. The voltage graph approach might help.
    #
    # Let me try the voltage graph approach with K_6 base.
    #
    # For m, the lifted graph has 6*m vertices. We want 6*m to be small, so
    # we want small m. But m must be large enough to avoid short cycles.
    #
    # Let's try m = 1, 2, 3, ... and see if we can find valid voltage assignments.
    
    deadline = time.time() + 120  # 120 seconds max
    
    # For the specific case k=5, g=9:
    if k != 5 or g != 9:
        # Fallback to L-U-W construction for other parameters
        return _luw_construct(k, g)
    
    # Try voltage graph over K_6
    base_n = 6
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_base_edges = len(base_edges)  # 15 edges
    
    # Enumerate non-backtracking closed walks of length 3..8 in K_6
    # A walk is a sequence of vertices v_0, ..., v_L where v_L = v_0,
    # consecutive vertices differ, and v_{i+1} != v_{i-1} (no backtracking).
    # We fix v_0 = 0 to avoid counting rotations.
    
    def gen_nonbacktracking_closed_walks(length):
        """Generate non-backtracking closed walks of given length in K_6, starting at 0."""
        if length < 3:
            return
        # v_0 = 0, v_L = 0
        # v_1 in 1..5
        # v_2 in 0..5 except v_1 and v_0=0 (no backtrack: v_2 != v_0)
        # Actually: v_2 != v_1 (edge) and v_2 != v_0 (no backtrack)
        # v_i for i >= 2: v_i != v_{i-1} and v_i != v_{i-2}
        # v_L = 0, so v_{L-1} != 0 and v_{L-1} != v_{L-2}
        
        def backtrack(v_idx, current, prev):
            if v_idx == length:
                # Check if we can close: current must connect to 0
                # and 0 != prev (no backtrack at closure)
                if current != 0 and prev != 0:
                    yield [0, 0, 0]  # placeholder, we'll build properly
                return
            if v_idx == length - 1:
                # Must go to 0
                if current != 0 and prev != 0:
                    yield [current]  # last step is to 0
                return
            
            for next_v in range(base_n):
                if next_v == current:
                    continue
                if next_v == prev:
                    continue
                # Continue
                yield from []
        
        # Better: iterative generation
        def walk_gen(pos, cur, prv, path):
            if pos == length:
                # Close the walk: need cur -> 0, and 0 != prv, cur != 0
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            if pos == length - 1:
                # Next must be 0
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            
            for nv in range(base_n):
                if nv == cur or nv == prv:
                    continue
                path.append(nv)
                yield from walk_gen(pos + 1, nv, cur, path)
                path.pop()
        
        for v1 in range(1, base_n):
            yield from walk_gen(1, v1, 0, [0, v1])
    
    # Collect all constraints: for each walk, we have a linear form in voltages
    # that must be non-zero mod m.
    
    constraints_by_length = {}
    for L in range(3, 9):
        walks = list(gen_nonbacktracking_closed_walks(L))
        # Deduplicate: two walks that are rotations or reversals give the same constraint
        # For simplicity, just collect all and dedupe later
        constraint_set = set()
        for walk in walks:
            # walk is [v_0, v_1, ..., v_L] with v_0 = v_L = 0
            # voltage sum = sum_{i=0}^{L-1} s_i * phi_{min(v_i,v_{i+1}), max(v_i,v_{i+1})}
            # where s_i = +1 if v_i < v_{i+1}, else -1
            terms = []
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                if u == v:
                    continue  # shouldn't happen
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                terms.append((eidx, sign))
            # Convert to a tuple for deduplication
            terms_tuple = tuple(terms)
            constraint_set.add(terms_tuple)
        constraints_by_length[L] = list(constraint_set)
    
    # Now try different m values and find valid voltage assignments
    # For each m, we need voltages in Z_m such that no constraint evaluates to 0.
    
    best_result = None
    
    for m in range(2, 300):
        if time.time() > deadline - 30:
            break
        
        n_vertices = base_n * m
        if best_result is not None and n_vertices >= best_result[0]:
            continue  # We want to minimize n, so skip if not smaller
        
        # Try random voltage assignments
        max_attempts = 5000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            voltages = [random.randint(0, m - 1) for _ in range(n_base_edges)]
            
            valid = True
            for L in range(3, 9):
                for terms in constraints_by_length[L]:
                    s = 0
                    for eidx, sign in terms:
                        s += sign * voltages[eidx]
                    if s % m == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                break
        
        if found:
            # Build the lifted graph
            graph_edges = []
            for u in range(base_n):
                for v in range(base_n):
                    if u == v:
                        continue
                    eu, ev = min(u, v), max(u, v)
                    eidx = edge_index[(eu, ev)]
                    for i in range(m):
                        vid1 = u * m + i
                        if u < v:
                            j = (i + voltages[eidx]) % m
                        else:
                            j = (i - voltages[eidx]) % m
                        vid2 = v * m + j
                        if vid1 < vid2:
                            graph_edges.append((vid1, vid2))
            
            best_result = (n_vertices, graph_edges)
            break  # Found a valid one for this m, go to next m (smaller n)
    
    if best_result is not None:
        return best_result
    
    # Fallback: use L-U-W construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
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
            yield 1 - side, tuple(y)
    
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
    return len(order), edges

# EVOLVE-BLOCK-END
