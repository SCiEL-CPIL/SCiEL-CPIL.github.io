import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 105.0

    def find_violation(adj):
        """Find a K3,3 violation. Returns (u, v, w) triple with >=3 common neighbors, or None."""
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        return (u, v, w)
        return None

    def repair(adj):
        """Remove edges to eliminate all K3,3 violations."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w = viol
            common = adj[u] & adj[v] & adj[w]
            cn_list = []
            c = common
            while c:
                lb = c & (-c)
                cn_list.append(lb.bit_length() - 1)
                c ^= lb
            
            if cn_list:
                cn = random.choice(cn_list)
                options = [(u, cn), (v, cn), (w, cn)]
                valid_options = [(a, b) for a, b in options if a != b and adj[a] & (1 << b)]
                if valid_options:
                    a, b = random.choice(valid_options)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    changed = True
                else:
                    tri_edges = [(u, v), (u, w), (v, w)]
                    valid_tri = [(a, b) for a, b in tri_edges if adj[a] & (1 << b)]
                    if valid_tri:
                        a, b = random.choice(valid_tri)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        changed = True
            else:
                break
        return adj

    def is_valid_after_add_fast(adj, u, v):
        """
        Fast check: only verify triples that include the new edge (u,v).
        A new K3,3 must have {u,v} in one partition and 3 common neighbors in the other.
        So we just need: |N(u) ∩ N(v)| >= 3 is NOT the criterion.
        Actually K3,3 means there exist sets A={a1,a2,a3}, B={b1,b2,b3} all edges between.
        Adding edge (u,v) can only create new K3,3 where (u,v) is an edge in the K3,3.
        So we need: does there exist a triple (x,y,z) such that {u,v} ∪ {x,y,z} forms K3,3?
        No - K3,3 has 6 vertices. The new edge (u,v) means u and v are in opposite partitions.
        So we need: exists partition {u, a, b} and {v, c, d} such that all 9 cross-edges exist.
        That means: a,b ∈ N(u)∩N(v), c,d ∈ N(u)∩N(v), and a,c,d ∈ N(v), b,c,d ∈ N(v) already satisfied.
        Wait, let me think more carefully.
        
        In K3,3, we have two sets of 3 with all 9 edges. If we add edge (u,v), 
        then u and v must be in opposite parts. So one part contains u plus 2 others, 
        other part contains v plus 2 others.
        The 2 others in u's part must be neighbors of both u and v (since they connect to v's part which includes v).
        The 2 others in v's part must be neighbors of both u and v (since they connect to u's part which includes u).
        So we need: at least 2 common neighbors of u and v that are also... 
        Actually: let CN = N(u) ∩ N(v). We need to pick 2 from CN for u's side and 2 from CN for v's side.
        But wait, the 2 on u's side connect to all 3 on v's side (including v), and the 2 on v's side connect to all 3 on u's side (including u).
        So if a is on u's side, a must be adjacent to v (yes, a ∈ CN) and to the 2 on v's side.
        If c is on v's side, c must be adjacent to u (yes, c ∈ CN) and to the 2 on u's side.
        So we need: |CN| >= 4, and among any 4 elements of CN, the cross-edges between the chosen 2+2 must exist.
        More precisely: we need to find a,b,c,d ∈ CN such that a-c, a-d, b-c, b-d all exist (and a-b, c-d don't matter for K3,3).
        Actually a,b are on same side so a-b edge doesn't matter. c,d same side, c-d doesn't matter.
        We need: for some distinct a,b,c,d in CN: a~c, a~d, b~c, b~d.
        
        Simpler: just check if there are >=3 common neighbors... no that's not right either.
        
        Let me just do the direct check but only for triples involving u or v:
        """
        # After adding (u,v), check all triples that include u or v
        # Triple (u, x, y): common neighbors = N(u)∩N(x)∩N(y) must have < 3
        # But N(u) now includes v, so new common neighbors could include v
        # v is in N(u)∩N(x)∩N(y) iff v~x and v~y, i.e., x,y ∈ N(v)
        # So for triple (u,x,y): check if v ∈ N(x)∩N(y) AND |old_common| >= 2 (so with v added, becomes 3)
        # OR |old_common| >= 3 (already violated, but we assume graph was valid before)
        # Since graph was valid before adding (u,v), |old_common| <= 2 for all triples.
        # New violation: |old_common| == 2 AND v ∈ N(x)∩N(y), making it 3.
        # Similarly for triples involving v.
        
        nu = adj[u]  # before adding v
        nv = adj[v]  # before adding u
        
        # Triples (u, x, y) for x < y, x,y != v
        # New common neighbor v: need v~x and v~y, i.e., x,y in nv
        # Old common: nu & adj[x] & adj[y] (without v since v not in nu yet... wait nu doesn't have v)
        # Actually adj[u] = nu doesn't include v yet. After adding, N(u) = nu | {v}.
        # So new common for (u,x,y) = (nu | {v}) & adj[x] & adj[y] = (nu & adj[x] & adj[y]) | ({v} & adj[x] & adj[y])
        # = old_common | (v if v~x and v~y else 0)
        # Violation if old_common.bit_count() + (1 if v~x and v~y else 0) >= 3
        # Since old was valid, old_common.bit_count() <= 2. So violation iff old_common.bit_count()==2 and v~x and v~y.
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                old_common = nu & adj[x] & adj[y]
                if old_common.bit_count() == 2 and (nv & (1 << x)) and (nv & (1 << y)):
                    return False
        
        # Triples (v, x, y) for x < y, x,y != u
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                old_common = nv & adj[x] & adj[y]
                if old_common.bit_count() == 2 and (nu & (1 << x)) and (nu & (1 << y)):
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """Greedily adds edges in random order using fast validity check."""
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not is_valid_after_add_fast(adj, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    def random_dense_graph(edge_prob):
        """Generate a random graph with given edge probability."""
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < edge_prob:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = [0] * n
    best_count = 0

    # Phase 1: Start from dense random graphs, repair, then greedy fill
    for trial in range(60):
        if time.time() > deadline:
            break
        
        if trial < 20:
            p = 0.7
        elif trial < 35:
            p = 0.6
        elif trial < 50:
            p = 0.5
        else:
            p = 0.4

        current_adj = random_dense_graph(p)
        current_adj = repair(current_adj)
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Phase 2: Local search - edge swaps and removals
    while time.time() < deadline - 5.0:
        current_adj = best_adj[:]
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove 1-2 random edges, then greedy fill
            num_removals = random.randint(1, 2)
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u] & (1 << v)]
            if not edges_list:
                break
            random.shuffle(edges_list)
            for u, v in edges_list[:num_removals]:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            current_count = greedy_fill(current_adj)
        elif move_type < 0.7:
            # 2-for-2 swap: remove 2 edges, add 2 different edges, if valid
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u] & (1 << v)]
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (current_adj[u] & (1 << v))]
            if len(edges_list) < 2 or len(non_edges) < 2:
                break
            e1, e2 = random.sample(edges_list, 2)
            n1, n2 = random.sample(non_edges, 2)
            # Remove e1, e2
            current_adj[e1[0]] &= ~(1 << e1[1])
            current_adj[e1[1]] &= ~(1 << e1[0])
            current_adj[e2[0]] &= ~(1 << e2[1])
            current_adj[e2[1]] &= ~(1 << e2[0])
            # Try adding n1, n2
            ok1 = True
            current_adj[n1[0]] |= (1 << n1[1])
            current_adj[n1[1]] |= (1 << n1[0])
            if not is_valid_after_add_fast(current_adj, n1[0], n1[1]):
                current_adj[n1[0]] &= ~(1 << n1[1])
                current_adj[n1[1]] &= ~(1 << n1[0])
                ok1 = False
            ok2 = True
            if ok1:
                current_adj[n2[0]] |= (1 << n2[1])
                current_adj[n2[1]] |= (1 << n2[0])
                if not is_valid_after_add_fast(current_adj, n2[0], n2[1]):
                    current_adj[n2[0]] &= ~(1 << n2[1])
                    current_adj[n2[1]] &= ~(1 << n2[0])
                    ok2 = False
            # If both added, keep; else revert to just having removed edges and do greedy
            if ok1 and ok2:
                current_count = count_edges(current_adj)
            else:
                # Revert the removals too and try greedy from original
                current_adj = best_adj[:]
                current_count = greedy_fill(current_adj)
        else:
            # Single edge swap: remove 1, add 1
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if current_adj[u] & (1 << v)]
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not (current_adj[u] & (1 << v))]
            if not edges_list or not non_edges:
                break
            e1 = random.choice(edges_list)
            n1 = random.choice(non_edges)
            current_adj[e1[0]] &= ~(1 << e1[1])
            current_adj[e1[1]] &= ~(1 << e1[0])
            current_adj[n1[0]] |= (1 << n1[1])
            current_adj[n1[1]] |= (1 << n1[0])
            if not is_valid_after_add_fast(current_adj, n1[0], n1[1]):
                current_adj[n1[0]] &= ~(1 << n1[1])
                current_adj[n1[1]] &= ~(1 << n1[0])
                # Revert removal
                current_adj[e1[0]] |= (1 << e1[1])
                current_adj[e1[1]] |= (1 << e1[0])
                current_count = count_edges(current_adj)
            else:
                current_count = count_edges(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Final greedy fill on best
    final_count = greedy_fill(best_adj)
    if final_count > best_count:
        best_count = final_count

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
