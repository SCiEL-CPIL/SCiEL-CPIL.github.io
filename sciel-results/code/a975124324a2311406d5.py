"""Dense Q3-free graph on 16 vertices via complement edge-removal (deletion) strategy with repair."""
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin
    n = 16
    total_edges = n * (n - 1) // 2  # 120

    def count_q3_in_subgraph(adj, vertex_set):
        """Count Q3 subgraphs within a specific set of 8 vertices using adj."""
        # vertex_set is a frozenset of 8 vertices
        if len(vertex_set) != 8:
            return 0
        verts = list(vertex_set)
        # Check all 8! / 6 = 840 bijections? No, too slow.
        # Better: check if the induced subgraph on these 8 vertices IS a Q3.
        # Q3 has 12 edges, each vertex degree 3.
        # Count edges in induced subgraph
        edge_count = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if adj[verts[i]] & (1 << verts[j]):
                    edge_count += 1
        if edge_count != 12:
            return 0
        # Check all degrees are 3
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and adj[verts[i]] & (1 << verts[j]):
                    deg += 1
            if deg != 3:
                return 0
        # Now verify it's actually a cube (not just a 3-regular 12-edge graph on 8 vertices)
        # A 3-regular graph on 8 vertices with 12 edges that is bipartite and connected is Q3.
        # Check bipartite: 2-color the subgraph
        color = {}
        from collections import deque
        start = verts[0]
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in verts:
                if v != u and adj[u] & (1 << v):
                    if v not in color:
                        color[v] = color[u] ^ 1
                        q.append(v)
                    elif color[v] == color[u]:
                        return 0  # Not bipartite
        # Check connected
        if len(color) != 8:
            return 0
        return 1  # It's Q3

    def count_q3_total(adj):
        """Count all Q3 subgraphs in the graph. Uses 8-subset enumeration."""
        from itertools import combinations
        count = 0
        for subset in combinations(range(n), 8):
            # Quick check: count edges in induced subgraph
            es = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[subset[i]] & (1 << subset[j]):
                        es += 1
            if es != 12:
                continue
            # Check degrees
            ok = True
            for i in range(8):
                deg = 0
                for j in range(8):
                    if i != j and adj[subset[i]] & (1 << subset[j]):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            # Check bipartite + connected
            from collections import deque
            color = {}
            start = subset[0]
            color[start] = 0
            dq = deque([start])
            valid = True
            while dq and valid:
                u = dq.popleft()
                for v in subset:
                    if v != u and adj[u] & (1 << v):
                        if v not in color:
                            color[v] = color[u] ^ 1
                            dq.append(v)
                        elif color[v] == color[u]:
                            valid = False
                            break
            if valid and len(color) == 8:
                count += 1
        return count

    def q3s_containing_edge(adj, u, v):
        """Count Q3 subgraphs that contain the specific edge (u,v)."""
        from itertools import combinations
        count = 0
        # The other 6 vertices must come from the remaining 14
        for subset in combinations([x for x in range(n) if x != u and x != v], 6):
            verts = [u, v] + list(subset)
            # Check edges
            es = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[verts[i]] & (1 << verts[j]):
                        es += 1
            if es != 12:
                continue
            # Check the specific edge (u,v) is present (it is by construction)
            # Check degrees
            ok = True
            for i in range(8):
                deg = 0
                for j in range(8):
                    if i != j and adj[verts[i]] & (1 << verts[j]):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            # Check bipartite + connected
            from collections import deque
            color = {}
            start = verts[0]
            color[start] = 0
            dq = deque([start])
            valid = True
            while dq and valid:
                uu = dq.popleft()
                for vv in verts:
                    if vv != uu and adj[uu] & (1 << vv):
                        if vv not in color:
                            color[vv] = color[uu] ^ 1
                            dq.append(vv)
                        elif color[vv] == color[uu]:
                            valid = False
                            break
            if valid and len(color) == 8:
                count += 1
        return count

    def is_q3_free(adj):
        """Check if graph is Q3-free."""
        from itertools import combinations
        for subset in combinations(range(n), 8):
            es = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[subset[i]] & (1 << subset[j]):
                        es += 1
            if es != 12:
                continue
            ok = True
            for i in range(8):
                deg = 0
                for j in range(8):
                    if i != j and adj[subset[i]] & (1 << subset[j]):
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            from collections import deque
            color = {}
            start = subset[0]
            color[start] = 0
            dq = deque([start])
            valid = True
            while dq and valid:
                uu = dq.popleft()
                for vv in subset:
                    if vv != uu and adj[uu] & (1 << vv):
                        if vv not in color:
                            color[vv] = color[uu] ^ 1
                            dq.append(vv)
                        elif color[vv] == color[uu]:
                            valid = False
                            break
            if valid and len(color) == 8:
                return False
        return True

    def creates_q3_fast(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def get_missing(adj):
        return [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]

    def get_existing(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append((u, v))
                mask -= bit
        return result

    def deletion_strategy(adj, max_time):
        """Iteratively delete edges that break the most Q3 subgraphs."""
        start_time = time.time()
        current_q3_count = count_q3_total(adj)
        
        while current_q3_count > 0 and time.time() < deadline - 5:
            if time.time() > start_time + max_time:
                break
            
            # Find the edge whose deletion removes the most Q3s
            existing = get_existing(adj)
            best_edge = None
            best_removed = -1
            
            # Sample edges to find good candidates (full enumeration is expensive)
            random.shuffle(existing)
            for u, v in existing[:40]:  # Check up to 40 random edges
                # Count Q3s containing this edge
                # This is expensive, so use a heuristic: check if removing it makes the graph Q3-free
                trial = adj[:]
                trial[u] &= ~(1 << v)
                trial[v] &= ~(1 << u)
                
                # Quick check: is trial Q3-free?
                # Use a fast check
                q3_count = count_q3_total(trial)
                removed = current_q3_count - q3_count
                if removed > best_removed:
                    best_removed = removed
                    best_edge = (u, v)
                
                if best_removed >= current_q3_count:
                    break
            
            if best_edge is None:
                # Try all edges
                for u, v in existing:
                    trial = adj[:]
                    trial[u] &= ~(1 << v)
                    trial[v] &= ~(1 << u)
                    q3_count = count_q3_total(trial)
                    removed = current_q3_count - q3_count
                    if removed > best_removed:
                        best_removed = removed
                        best_edge = (u, v)
            
            if best_edge is None or best_removed <= 0:
                break
            
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_q3_count -= best_removed
        
        return adj

    def repair_pass(adj, max_time):
        """Try to re-add edges that don't create Q3 (repair the over-deletion)."""
        start_time = time.time()
        added = 0
        while time.time() < deadline - 5 and time.time() < start_time + max_time:
            missing = get_missing(adj)
            if not missing:
                break
            
            # Sort by some heuristic: prefer edges between low-degree vertices
            missing.sort(key=lambda e: (adj[e[0]].bit_count() + adj[e[1]].bit_count()))
            
            progress = False
            for u, v in missing:
                if time.time() > deadline - 5 or time.time() > start_time + max_time:
                    break
                if not (adj[u] & (1 << v)):
                    if not creates_q3_fast(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added += 1
                        progress = True
            
            if not progress:
                break
        
        return adj, added

    def local_search_deletion(adj, max_time):
        """Local search in the deletion space: try swapping deletions."""
        start_time = time.time()
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        for _ in range(30):
            if time.time() > deadline - 5 or time.time() > start_time + max_time:
                break
            
            # Try: remove one edge, then repair
            existing = get_existing(adj)
            if not existing:
                break
            
            u_rem, v_rem = random.choice(existing)
            trial = adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            
            # Repair the trial
            trial, _ = repair_pass(trial, max_time=2)
            
            new_count = count_edges(trial)
            if new_count > best_count:
                best_count = new_count
                best_adj = trial[:]
                adj = trial[:]
        
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1

    # Start from complete graph and delete
    # Multiple strategies:
    # 1. Pure deletion from K16
    # 2. Deletion from structured dense graphs
    
    # Strategy 1: Pure deletion from K16
    adj_k16 = [(1 << n) - 2] * n  # Each vertex connected to all others
    for i in range(n):
        adj_k16[i] &= ~(1 << i)  # Remove self-loops
    
    adj1 = deletion_strategy(adj_k16[:], max_time=30)
    adj1, _ = repair_pass(adj1, max_time=15)
    adj1, cnt1 = local_search_deletion(adj1, max_time=15)
    
    if cnt1 > best_count:
        best_count = cnt1
        best_adj = adj1[:]

    # Strategy 2: Deletion from K8,8
    adj_k88 = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_k88[u] |= (1 << v)
            adj_k88[v] |= (1 << u)
    
    adj2 = deletion_strategy(adj_k88[:], max_time=20)
    adj2, _ = repair_pass(adj2, max_time=15)
    adj2, cnt2 = local_search_deletion(adj2, max_time=15)
    
    if cnt2 > best_count:
        best_count = cnt2
        best_adj = adj2[:]

    # Strategy 3: Deletion from Turan T(16,4)
    adj_turan = [0] * n
    parts = [[i for i in range(4 * k, 4 * k + 4)] for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    
    adj3 = deletion_strategy(adj_turan[:], max_time=20)
    adj3, _ = repair_pass(adj3, max_time=15)
    adj3, cnt3 = local_search_deletion(adj3, max_time=15)
    
    if cnt3 > best_count:
        best_count = cnt3
        best_adj = adj3[:]

    # Strategy 4: Random dense init then deletion
    for seed_try in range(3):
        if time.time() > deadline - 20:
            break
        
        adj_rand = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.7:
                    adj_rand[u] |= (1 << v)
                    adj_rand[v] |= (1 << u)
        
        adj_r = deletion_strategy(adj_rand[:], max_time=15)
        adj_r, _ = repair_pass(adj_r, max_time=10)
        adj_r, cnt_r = local_search_deletion(adj_r, max_time=10)
        
        if cnt_r > best_count:
            best_count = cnt_r
            best_adj = adj_r[:]

    # Final polish: aggressive repair
    if time.time() < deadline - 3:
        best_adj, _ = repair_pass(best_adj, max_time=10)
        best_count = count_edges(best_adj)

    # Verify Q3-free
    if not is_q3_free(best_adj):
        # Force repair
        best_adj, _ = repair_pass(best_adj, max_time=5)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
