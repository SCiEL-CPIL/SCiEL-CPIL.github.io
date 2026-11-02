"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with carefully chosen voltages.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is at least 10 if the voltage assignment avoids short cycles.
"""
import time
import sys
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    # Vertex (side, i, k) where side=0 for A, side=1 for B, i in 0..3, k in 0..39.
    # Edge from (0, i, k) to (1, j, (k + volt[i][j]) % 40).
    
    # A lifted cycle of length L corresponds to a closed walk of length L in K_{4,4}
    # with total voltage sum = 0 mod 40.
    # For girth >= 10, we need no closed walk of length 4, 6, or 8 with voltage sum 0 mod 40.
    
    # Closed walks of length 4 in K_{4,4}: a_i -> b_j -> a_k -> b_l -> a_i, i!=k, j!=l
    # Voltage: volt[i][j] - volt[k][j] + volt[k][l] - volt[i][l] mod 40
    # (signs: forward on (a_i,b_j), backward on (a_k,b_j), forward on (a_k,b_l), backward on (a_i,b_l))
    
    # Closed walks of length 6: a_i -> b_j -> a_k -> b_l -> a_m -> b_n -> a_i
    # with i!=k, k!=m, m!=i (not all same) and j!=l, l!=n, n!=j
    # Voltage: volt[i][j] - volt[k][j] + volt[k][l] - volt[m][l] + volt[m][n] - volt[i][n] mod 40
    
    # Closed walks of length 8: similar pattern.
    
    # Strategy: search for voltage matrix volt[4][4] in Z_40 such that:
    # 1. No 4-closed-walk has sum 0: volt[i][j] - volt[k][j] + volt[k][l] - volt[i][l] != 0 for all i!=k, j!=l
    # 2. No 6-closed-walk has sum 0
    # 3. No 8-closed-walk has sum 0
    
    # Condition 1 is equivalent to: for all i!=k, the vector (volt[i][0]-volt[k][0], ..., volt[i][3]-volt[k][3])
    # has all distinct entries (no two column differences are equal).
    # More precisely: volt[i][j] - volt[i][l] != volt[k][j] - volt[k][l] for all i!=k, j!=l.
    
    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges
    
    def _check_4cycle_condition(voltages):
        """Check no 4-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for k in range(4):
                if i == k:
                    continue
                for j in range(4):
                    for l in range(4):
                        if j == l:
                            continue
                        s = (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % 40
                        if s == 0:
                            return False
        return True
    
    def _check_6cycle_condition(voltages):
        """Check no 6-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if k == i:
                        continue
                    for l in range(4):
                        if l == j:
                            continue
                        for m in range(4):
                            if m == k or m == i:
                                continue
                            for n in range(4):
                                if n == l or n == j:
                                    continue
                                s = (voltages[i][j] - voltages[k][j] + voltages[k][l] 
                                     - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                                if s == 0:
                                    return False
        return True
    
    def _check_8cycle_condition(voltages):
        """Check no 8-closed-walk in base has voltage 0 mod 40."""
        # Simple 8-cycles in K_{4,4} use all 4 A-vertices and all 4 B-vertices.
        # Fix a_{i0}=0, a_{i1}=1 (by symmetry) to reduce permutations.
        # Then permute the remaining 2 A-vertices and all 4 B-vertices.
        import itertools
        # Fix first two A vertices to break rotational symmetry
        for perm_a_rest in itertools.permutations([2, 3]):
            for perm_b in itertools.permutations(range(4)):
                # Full A sequence: [0, 1, perm_a_rest[0], perm_a_rest[1]]
                perm_a = [0, 1, perm_a_rest[0], perm_a_rest[1]]
                # Voltage sum for cycle: a0->b0->a1->b1->a2->b2->a3->b3->a0
                s = 0
                for idx in range(4):
                    a_cur = perm_a[idx]
                    b_cur = perm_b[idx]
                    a_next = perm_a[(idx + 1) % 4]
                    # Forward edge a_cur -> b_cur: +volt[a_cur][b_cur]
                    # Backward edge a_next -> b_cur: -volt[a_next][b_cur]
                    s += voltages[a_cur][b_cur]
                    s -= voltages[a_next][b_cur]
                if s % 40 == 0:
                    return False
        # Also need to check cycles where the A ordering is different
        # The above fixes A order as 0,1,2,3 (with 2,3 swapped). But other A orderings exist.
        # Actually, any 8-cycle corresponds to a cyclic permutation of A and B.
        # By fixing a0=0 and a1=1, we've fixed 2 of 4 A positions. The remaining 2 can be in any order.
        # But we also need to consider that the B vertices can be in any order.
        # The above covers all cases where A order starts with 0,1. By rotational symmetry of the cycle,
        # this covers all distinct 8-cycles.
        return True
    
    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For girth >= g, no non-backtracking closed walk of length < g exists.
        # Non-backtracking walk: (v0, v1, v2, ..., vL) where vi != vi-2 for all i >= 2.
        # A closed non-backtracking walk of length L < g would be a cycle of length <= L < g.
        # Actually, a non-backtracking closed walk of length L is a cycle of length L
        # (if L >= 3 and the walk is non-backtracking and closed, it's a simple cycle of length L
        # or contains one).
        
        # Efficient check: for each vertex, enumerate all non-backtracking walks of length up to g-1
        # and check if any returns to the start.
        
        max_len = g - 1  # we check walks of length 1..g-1
        
        for start_v in range(n):
            if time.time() > deadline:
                return False
            # State: (current_vertex, previous_vertex)
            # Walk of length 1: start_v -> w for w in adj[start_v]
            # Walk of length 2: start_v -> w -> x where x != start_v
            # Walk of length L: ... -> y -> z where z != y's predecessor
            
            # BFS over (vertex, prev) states
            # Level 0: start at (start_v, -1) meaning "just started"
            # Actually, let's track walks explicitly.
            
            # walks_of_length_l: set of (current, prev) pairs
            # Length 1: for each w in adj[start_v]: (w, start_v)
            # Length 2: for each (w, start_v): for each x in adj[w] where x != start_v: (x, w)
            # ...
            # A closed walk of length L exists if start_v appears as 'current' at level L
            # with prev != start_v (non-backtracking) -- actually we need the walk to close back to start_v.
            
            # Let me redo: a non-backtracking walk of length L starting and ending at start_v:
            # (start_v, v1, v2, ..., v_{L-1}, start_v) where vi != v_{i-2} for all valid i.
            # Equivalently: v1 in adj(start_v), v2 in adj(v1) \ {start_v}, ..., v_{L-1} in adj(v_{L-2}) \ {v_{L-3}}, start_v in adj(v_{L-1}) \ {v_{L-2}}.
            
            # For L=4: start_v -> a -> b -> c -> start_v, with a!=start_v (auto), b!=start_v, c!=a, start_v!=b.
            #   a in adj(start_v), b in adj(a)\{start_v}, c in adj(b)\{a}, start_v in adj(c)\{b}.
            
            # Let's do BFS tracking (current, prev) and depth.
            # At depth d, we have states (v, p) meaning we're at v, came from p.
            # We want to know if at depth L, we can reach (start_v, p) for some p != the vertex at depth L-2.
            # Actually, if we reach (start_v, p) at depth L, the walk is start_v -> ... -> p -> start_v.
            # Non-backtracking requires p != start_v (which is true since p is a neighbor of start_v and start_v != p)
            # and also the step before p must not be start_v... this gets complicated.
            
            # Simpler: just check if start_v is reachable in exactly L steps via non-backtracking walk.
            # Use DP: dp[d][v][p] = True if there's a non-backtracking walk of length d from start_v to v, last step from p.
            
            # For efficiency, use sets.
            # states[d] = set of (v, p) pairs reachable by non-backtracking walk of length d from start_v.
            # states[1] = {(w, start_v) for w in adj(start_v)}
            # states[d] = {(w, v) for (v, p) in states[d-1] for w in adj(v) if w != p}
            # Closed walk of length L exists if (start_v, p) in states[L] for some p.
            
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))
            
            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break
        
        return True
    
    # Search for valid voltages
    candidate_voltages = None
    
    # First try the simplest: volt[i][j] = c*i*j mod 40 with various c
    for c in range(1, 40):
        if c % 10 == 0:
            continue  # fails condition 1
        voltages = [[(c * i * j) % 40 for j in range(4)] for i in range(4)]
        if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
            candidate_voltages = voltages
            break
    
    if candidate_voltages is None:
        # Try volt[i][j] = (a*i + b*j + c*i*j) mod 40
        for c in [1, 3, 7, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39]:
            for a in range(0, 40, 2):
                for b in range(0, 40, 2):
                    if time.time() > deadline - 30:
                        break
                    voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
                    if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                        candidate_voltages = voltages
                        break
                if candidate_voltages:
                    break
            if candidate_voltages:
                break
    
    if candidate_voltages is None:
        # Random search with more attempts
        for attempt in range(20000):
            if time.time() > deadline - 30:
                break
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                candidate_voltages = voltages
                break
    
    if candidate_voltages is not None:
        n, edges = _build_lift(candidate_voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
        # If verification fails, the voltage assignment was insufficient
        # Try more random searches
        for attempt in range(10000):
            if time.time() > deadline - 20:
                break
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                n2, edges2 = _build_lift(voltages)
                if _verify_girth_exact(n2, edges2, g, deadline):
                    return n2, edges2
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
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
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
