import time
import random
from collections import defaultdict, deque
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Calculate Moore Bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # 2. Fallback: LUW Construction
    best_n, best_edges = construct_luw(k, g)
    
    # 3. Try Voltage Graph / Lift Construction with optimized DP check
    try:
        n_lift, edges_lift = construct_voltage_lift_optimized(k, g, deadline)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass
    
    # 4. Try Greedy Tree Construction (Moore bound) - likely to fail for g=9, k=5
    # but kept for completeness if it ever works
    n_target = moore_bound
    if k % 2 == 1 and n_target % 2 != 0:
        n_target += 1
        
    try:
        n_greedy, edges_greedy = construct_greedy_tree(k, g, n_target)
        if n_greedy is not None and n_greedy < best_n:
            best_n = n_greedy
            best_edges = edges_greedy
    except Exception:
        pass
        
    return best_n, best_edges

def construct_luw(k, g):
    # Original LUW logic, kept as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

def construct_voltage_lift_optimized(k, g, deadline):
    """
    Optimized voltage graph lift search using DP for girth check.
    """
    
    def make_circulant_k_reg(n, k):
        """Generate a circulant k-regular graph on n vertices."""
        if k % 2 != 0:
            if n % 2 == 0:
                if (k - 1) // 2 + 1 > n // 2:
                    return None
                diffs = list(range(1, (k - 1) // 2 + 1)) + [n // 2]
            else:
                if k >= n:
                    return None
                diffs = list(range(1, (k + 1) // 2 + 1))
                if len(diffs) * 2 > n - 1:
                    return None
        else:
            if k >= n:
                return None
            diffs = list(range(1, k // 2 + 1))
            if len(diffs) * 2 > n - 1:
                return None
        
        edges = set()
        for d in diffs:
            for v in range(n):
                u = (v + d) % n
                edges.add((min(v, u), max(v, u)))
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        if all(len(adj[v]) == k for v in range(n)):
            return list(edges), adj
        return None, None

    def make_random_k_reg(n, k):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None, None
        for _ in range(100):
            stubs = []
            for v in range(n):
                stubs.extend([v] * k)
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u,v), max(u,v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            if valid:
                adj = defaultdict(set)
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                return list(edges), adj
        return None, None

    def check_girth_dp(base_adj, base_edges, voltages, m, g):
        """
        Check if cover has girth >= g using DP on non-backtracking walks.
        Returns True if no closed walk of length < g has voltage sum 0.
        """
        n_base = len(base_adj)
        
        # Precompute neighbors list for faster iteration
        neighbors = [list(base_adj[v]) for v in range(n_base)]
        
        # voltages[u][v] is the voltage from u to v
        # We need to ensure voltages are defined for all directed edges in base_adj
        # voltages is a dict (u, v) -> val
        
        for L in range(3, g):
            # S_l[u][v] = set of voltage sums for non-backtracking walks of length l ending at u, coming from v
            # We only care about walks of length L.
            # Initialize for L=1
            # S_1[u][v] = {vol(v, u)} for v in N(u)
            
            # We can do iterative DP
            # prev_S is dict: (u, v) -> set of sums
            # curr_S is dict: (u, w) -> set of sums
            
            # For L=1, the walk is just the edge.
            # But we need to build up to L.
            
            # Let's use a dictionary of sets.
            # Key: (current, prev)
            # Value: set of voltage sums
            
            # L=1
            curr_S = {}
            for u in range(n_base):
                for v in neighbors[u]:
                    # Walk v -> u
                    vol = voltages[(v, u)]
                    curr_S[(u, v)] = {vol}
            
            # If L==1, check closed walks? No, min cycle length is 3 in simple graph.
            # We start checking for closed walks from L=3.
            
            if L == 1:
                continue
            
            # L=2
            next_S = {}
            for (u, v), sums in curr_S.items():
                # Extend walk ... -> v -> u to ... -> v -> u -> w
                # Non-backtracking: w != v
                for w in neighbors[u]:
                    if w == v:
                        continue
                    vol = voltages[(u, w)]
                    new_sums = { (s + vol) % m for s in sums }
                    key = (w, u)
                    if key in next_S:
                        next_S[key].update(new_sums)
                    else:
                        next_S[key] = new_sums
            
            if L == 2:
                # Check for closed walks of length 2? Impossible in simple graph.
                # But just in case:
                for (u, v), sums in next_S.items():
                    if u == v: # Should not happen in simple graph
                        if 0 in sums:
                            return False
                curr_S = next_S
                continue
            
            # L >= 3
            # We have curr_S for length L-1.
            # We want to compute length L.
            
            # Actually, let's generalize the loop.
            # We already did L=1 to get curr_S.
            # If L==1, we are done for this L iteration (but we skip check).
            # If L==2, we computed next_S.
            
            # Let's restructure to avoid duplicate logic.
            
            # Reset for each L
            # We need S_L.
            # Start with S_1.
            
            S_prev = {}
            for u in range(n_base):
                for v in neighbors[u]:
                    S_prev[(u, v)] = {voltages[(v, u)]}
            
            if L == 1:
                pass # No check needed for L=1 in simple graph context for girth>=3
            else:
                S_curr = S_prev
                for step in range(2, L + 1):
                    S_next = {}
                    for (u, v), sums in S_curr.items():
                        # u is current, v is prev.
                        # Next node w != v.
                        for w in neighbors[u]:
                            if w == v:
                                continue
                            vol = voltages[(u, w)]
                            new_sums = { (s + vol) % m for s in sums }
                            key = (w, u)
                            if key in S_next:
                                S_next[key].update(new_sums)
                            else:
                                S_next[key] = new_sums
                    S_curr = S_next
                
                # S_curr now holds S_L
                # Check for closed walks of length L
                # A closed walk of length L starting at x ends at x.
                # The state is (current=x, prev=y).
                # The walk is x -> ... -> y -> x.
                # The voltage sum in S_L[x][y] includes the edge y->x?
                # Yes, S_L[u][v] includes the voltage of the last edge v->u.
                # So if we have a state (x, y) in S_curr, the walk ends at x coming from y.
                # For it to be closed, it must have started at x.
                # The DP doesn't track the start.
                # However, a closed walk of length L is a cycle.
                # Any cycle of length L in the cover corresponds to a closed walk of length L in the base with voltage sum 0.
                # Does S_L[x][y] containing 0 imply a closed walk?
                # No, it implies a walk from *some* start to x.
                # We need to ensure the start is x.
                
                # This DP approach as stated does NOT track the start vertex.
                # It tracks the end and the immediate predecessor.
                # This is insufficient to detect *closed* walks unless we track the start.
                
                # Alternative: For each start vertex s, run a DP that tracks (current, prev, start).
                # State: (current, prev, start).
                # This increases state space by factor n_base.
                # n_base ~ 10, k=5. States ~ 10*5*10 = 500.
                # Sets of size m.
                # This is feasible.
                
                # Let's re-implement check_girth_dp with start tracking.
                
                # Actually, we can just check if the identity is in the product of matrices?
                # Let's stick to the explicit start tracking DP.
                
                # We'll do this inside the L loop, but it's expensive to restart.
                # We can do a BFS from all starts simultaneously?
                # State: (current, prev, start).
                # Initialize: for all s, for all v in N(s), state (v, s, s) with sum vol(s, v).
                
                # We'll do this check for each L.
                
                # To save time, we only do this if L is small enough or we are close to deadline.
                # But we must be correct.
                
                # Let's optimize the inner loop.
                
                # We'll create a helper function for the start-tracking DP.
                
                def has_closed_walk_length_L(L_check):
                    # S[step][current][prev][start] = set of sums
                    # This is too deep. Use dict: (current, prev, start) -> set of sums
                    
                    # Init for step 1
                    S = {}
                    for s in range(n_base):
                        for v in neighbors[s]:
                            vol = voltages[(s, v)]
                            key = (v, s, s)
                            S[key] = {vol}
                    
                    if L_check == 1:
                        # Closed walk of length 1? No.
                        return False
                    
                    for step in range(2, L_check + 1):
                        S_next = {}
                        for (u, v, s), sums in S.items():
                            # u is current, v is prev, s is start.
                            # Next w != v.
                            for w in neighbors[u]:
                                if w == v:
                                    continue
                                vol = voltages[(u, w)]
                                new_sums = { (x + vol) % m for x in sums }
                                key = (w, u, s)
                                if key in S_next:
                                    S_next[key].update(new_sums)
                                else:
                                    S_next[key] = new_sums
                        S = S_next
                        
                    # Check for closed walks
                    for (u, v, s), sums in S.items():
                        if u == s:
                            if 0 in sums:
                                return True
                    return False
                
                if has_closed_walk_length_L(L):
                    return False
        
        return True

    best_result = None
    
    # Try different base graph sizes
    # We want small N = m * n_base.
    # So try small n_base and small m.
    
    for base_n in range(10, 30):
        if time.time() > deadline:
            break
            
        base_edges, base_adj = make_circulant_k_reg(base_n, k)
        if base_edges is None:
            base_edges, base_adj = make_random_k_reg(base_n, k)
        if base_edges is None:
            continue
        
        if not all(len(base_adj[v]) == k for v in range(base_n)):
            continue
            
        # Try small m
        # m must be such that N = m * base_n is reasonable.
        # Also m should be large enough to avoid short cycles.
        # Typically m >= k is a good start.
        
        for m in range(2, 50):
            if time.time() > deadline:
                break
                
            total_n = m * base_n
            
            # Optional: Skip if total_n > best_n
            if best_result is not None and total_n >= best_result[0]:
                # We can break if m is increasing and base_n is fixed?
                # Yes, for fixed base_n, larger m means larger total_n.
                # But we want the *smallest* total_n.
                # So if we find a valid one, we can break for this base_n?
                # No, we want to find *any* valid one with total_n < best_n.
                # If we find one, we update best.
                # We can break the m loop if we found a valid one?
                # No, maybe a smaller m works? We iterate m from small to large.
                # So the first valid m is the smallest for this base_n.
                # So we can break the m loop after finding one valid m.
                pass
            
            # Try a few random voltage assignments
            found_valid = False
            for volt_attempt in range(10):
                if time.time() > deadline:
                    break
                    
                voltages = {}
                for (u, v) in base_edges:
                    val = random.randint(0, m - 1)
                    voltages[(u, v)] = val
                    voltages[(v, u)] = (-val) % m
                    
                if check_girth_dp(base_adj, base_edges, voltages, m, g):
                    # Valid
                    cover_edges = []
                    for (u, v) in base_edges:
                        vu = voltages[(u, v)]
                        for t in range(m):
                            u_cover = u * m + t
                            v_cover = v * m + (t + vu) % m
                            cover_edges.append((u_cover, v_cover))
                    
                    if best_result is None or total_n < best_result[0]:
                        best_result = (total_n, cover_edges)
                    found_valid = True
                    break # Found valid for this m, try next base_n or larger m?
                    # Since we iterate m from small to large, this is the smallest m for this base_n.
                    # So we can break the m loop.
            
            if found_valid:
                # We found the smallest m for this base_n.
                # Continue to next base_n to see if a smaller total_n is possible.
                # (e.g. base_n=10, m=20 -> 200. base_n=11, m=15 -> 165).
                # So we don't break the outer loop.
                pass
            
    return best_result

def construct_greedy_tree(k, g, n_target):
    """
    Try to construct a k-regular graph of girth g on exactly n_target vertices.
    """
    r = (g - 1) // 2
    moore = 1
    for i in range(1, r + 1):
        moore += k * ((k - 1) ** (i - 1))
    
    if n_target != moore:
        return None, None
    
    parent = [-1] * n_target
    level = [0] * n_target
    branch = [0] * n_target
    
    parent[0] = -1
    level[0] = 0
    
    next_id = 1
    for i in range(k):
        parent[next_id] = 0
        level[next_id] = 1
        branch[next_id] = i
        next_id += 1
        
    for l in range(2, r + 1):
        nodes_at_prev_level = [v for v in range(next_id) if level[v] == l-1]
        for v in nodes_at_prev_level:
            for j in range(k - 1):
                parent[next_id] = v
                level[next_id] = l
                branch[next_id] = branch[v]
                next_id += 1
                
    if next_id != n_target:
        return None, None
    
    leaves = [v for v in range(n_target) if level[v] == r]
    
    adj = defaultdict(set)
    for v in range(1, n_target):
        u = parent[v]
        adj[u].add(v)
        adj[v].add(u)
        
    deg_needed = {v: k - len(adj[v]) for v in range(n_target)}
    
    random.shuffle(leaves)
    
    for attempt in range(50):
        edges_to_add = []
        used_edges = set()
        success = True
        
        stubs = []
        for v in leaves:
            for _ in range(deg_needed[v]):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                success = False
                break
            if branch[u] == branch[v]:
                success = False
                break
            if v in adj[u]:
                success = False
                break
            if (min(u,v), max(u,v)) in used_edges:
                success = False
                break
            used_edges.add((min(u,v), max(u,v)))
            edges_to_add.append((u, v))
            
        if not success:
            continue
            
        final_edges = []
        for v in range(1, n_target):
            final_edges.append((parent[v], v))
        final_edges.extend(edges_to_add)
        
        deg_check = defaultdict(int)
        for u, v in final_edges:
            deg_check[u] += 1
            deg_check[v] += 1
        if all(deg_check[v] == k for v in range(n_target)):
            return n_target, final_edges
            
    return None, None
