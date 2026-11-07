import time
import sys
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Start with K_7,8
    # A: 0-6, B: 7-14
    base_edges = set()
    for u in range(7):
        for v in range(7, 15):
            base_edges.add((u, v))

    # All possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    missing_edges = list(all_edges - base_edges)

    # Fast validity check using adjacency matrix and degree array
    def build_state(edge_set):
        """Build adjacency and degree state from edge set."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def is_valid_fast(adj, deg, edge_set):
        """Check validity given precomputed adj and deg."""
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def check_add_edge(adj, deg, edge_set, new_edge):
        """Check if adding new_edge maintains validity. Only check paths involving new_edge."""
        u, v = new_edge
        # New degree values
        deg_u_new = deg[u] + 1
        deg_v_new = deg[v] + 1
        
        # Paths where new_edge is the middle edge (u-v): s-u-v-t
        # s in adj[u] (s != v), t in adj[v] (t != u, t != s)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                # Check if deg[s] == deg[t] in the NEW graph
                # deg[s] unchanged unless s == u or s == v (but s != v, and s != u since s is neighbor of u not v... wait s could be v but we skip that)
                # Actually s is a neighbor of u, s != v. s could equal u? No, no loops.
                # deg[s] is unchanged (s is not u or v)
                # deg[t] is unchanged (t is not u or v)
                if deg[s] == deg[t]:
                    return False
        
        # Paths where new_edge is an endpoint edge: s-u-v-t where new_edge is s-u or v-t
        # Case 1: new_edge is (s,u), so s=u, and we need t-v as the other endpoint... 
        # Actually let's think differently. Adding edge (u,v) creates new paths:
        # 1. Paths where (u,v) is the middle edge: s-u-v-t (already checked above)
        # 2. Paths where (u,v) is the first edge: u-v-x-y, need deg(u) != deg(y)
        # 3. Paths where (u,v) is the last edge: x-y-u-v, need deg(x) != deg(v)
        
        # Case 2: u-v-x-y, x in adj[v] (x != u), y in adj[x] (y != v, y != u)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                # deg(u) in new graph is deg_u_new, deg(y) unchanged
                if deg_u_new == deg[y]:
                    return False
        
        # Case 3: x-y-u-v, y in adj[u] (y != v), x in adj[y] (x != u, x != v)
        for y in adj[u]:
            if y == v:
                continue
            for x in adj[y]:
                if x == u or x == v:
                    continue
                # deg(x) unchanged, deg(v) in new graph is deg_v_new
                if deg[x] == deg_v_new:
                    return False
        
        return True

    best_edges = list(base_edges)
    best_count = len(base_edges)

    def local_improve_fast(edge_set):
        """Try to add any missing edge that maintains validity, repeat until no more can be added."""
        adj, deg = build_state(edge_set)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for e in missing_edges:
                if e in edge_set:
                    continue
                if time.time() > deadline:
                    break
                if check_add_edge(adj, deg, edge_set, e):
                    edge_set.add(e)
                    # Update adj and deg
                    u, v = e
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    improved = True
        return edge_set

    # Multiple restarts with different orderings
    rng = random.Random(42)
    num_restarts = 50

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Different ordering strategies
        strategy = restart % 5
        if strategy == 0:
            # Intra-A first, then intra-B, then cross
            intra_a = [e for e in missing_edges if e[0] < 7 and e[1] < 7]
            intra_b = [e for e in missing_edges if e[0] >= 7 and e[1] >= 7]
            cross = [e for e in missing_edges if not ((e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7))]
            missing_sorted = intra_a + intra_b + cross
        elif strategy == 1:
            # Intra-B first, then intra-A, then cross
            intra_a = [e for e in missing_edges if e[0] < 7 and e[1] < 7]
            intra_b = [e for e in missing_edges if e[0] >= 7 and e[1] >= 7]
            cross = [e for e in missing_edges if not ((e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7))]
            missing_sorted = intra_b + intra_a + cross
        elif strategy == 2:
            # Random shuffle
            missing_sorted = missing_edges[:]
            rng.shuffle(missing_sorted)
        elif strategy == 3:
            # Cross first
            intra_a = [e for e in missing_edges if e[0] < 7 and e[1] < 7]
            intra_b = [e for e in missing_edges if e[0] >= 7 and e[1] >= 7]
            cross = [e for e in missing_edges if not ((e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7))]
            rng.shuffle(cross)
            missing_sorted = cross + intra_a + intra_b
        else:
            # Random with intra preference
            intra = [e for e in missing_edges if (e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7)]
            cross = [e for e in missing_edges if not ((e[0] < 7 and e[1] < 7) or (e[0] >= 7 and e[1] >= 7))]
            rng.shuffle(intra)
            rng.shuffle(cross)
            # Interleave
            missing_sorted = []
            i, j = 0, 0
            while i < len(intra) or j < len(cross):
                if i < len(intra):
                    missing_sorted.append(intra[i])
                    i += 1
                if j < len(cross):
                    missing_sorted.append(cross[j])
                    j += 1

        # Greedy pass with fast check
        current_set = set(base_edges)
        adj, deg = build_state(current_set)
        for e in missing_sorted:
            if time.time() > deadline:
                break
            if e in current_set:
                continue
            if check_add_edge(adj, deg, current_set, e):
                current_set.add(e)
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1

        # Local improvement
        current_set = local_improve_fast(current_set)

        if len(current_set) > best_count:
            best_count = len(current_set)
            best_edges = list(current_set)

        # If we've used more than 70% of time, stop restarting
        elapsed = time.time() - start_time
        if elapsed > 0.7 * (deadline - start_time):
            break

    # Final local improvement on best result
    if time.time() < deadline - 2:
        best_set = set(best_edges)
        best_set = local_improve_fast(best_set)
        best_count = len(best_set)
        best_edges = list(best_set)

    # Convert to required format
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return final_edges
