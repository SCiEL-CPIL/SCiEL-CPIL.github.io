import time
import random as _rnd
import math

def construct():
    t_start = time.time()
    deadline = t_start + 115.0

    def generate_edges(n, p, seed):
        rng = _rnd.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            PX = P | X
            if PX:
                best_p = -1
                best_cnt = -1
                pp = PX
                while pp:
                    lb = pp & (-pp)
                    p = lb.bit_length() - 1
                    cnt = (adj[p] & P).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_p = p
                    pp ^= lb
                candidates = P & ~(adj[best_p])
            else:
                candidates = P

            cp = candidates
            while cp:
                lb = cp & (-cp)
                u = lb.bit_length() - 1
                cp ^= lb
                bk(R | lb, P & adj[u], X & adj[u])
                P ^= lb
                X |= lb

        bk(0, all_mask, 0)
        return cliques

    def filter_cliques(cliques, min_size=2):
        return [c for c in cliques if len(c) >= min_size]

    def greedy_hitting_set(cliques, n):
        uncovered = set(range(len(cliques)))
        vertex_cliques = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)

        hit = set()
        while uncovered:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            hit.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return hit

    def solve_instance(n, p, seed, time_limit):
        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            return []

        num_cliques = len(cliques)
        
        # Precompute clique masks (vertex sets)
        clique_masks = [0] * num_cliques
        for i, c in enumerate(cliques):
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks[i] = mask

        # Precompute vertex -> cliques bitmask
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)

        # Precompute Disjointness Mask for each clique
        # disjoint_mask[i] has bit j set if clique i and clique j are disjoint
        disjoint_mask = [0] * num_cliques
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                if not (clique_masks[i] & clique_masks[j]):
                    disjoint_mask[i] |= (1 << j)
                    disjoint_mask[j] |= (1 << i)

        # Lower Bound: Greedy Disjoint Packing
        def lb_greedy(remaining_mask):
            count = 0
            rem = remaining_mask
            while rem:
                # Pick the clique with the smallest number of disjoint neighbors in rem?
                # Or just the first one?
                # Let's pick the one that allows the most packing (smallest intersection)
                # For speed, just pick the lowest index bit
                lb = rem & (-rem)
                i = lb.bit_length() - 1
                count += 1
                # Remove i and all cliques intersecting i (i.e., keep only disjoint ones? No, we remove intersecting ones from consideration for *this* step, but actually we remove i and all that *intersect* i because we can't pick them. We *can* pick disjoint ones later.)
                # Wait, the greedy algorithm for Max Disjoint Set:
                # Pick a clique C. Add to set. Remove C and all cliques that intersect C from the pool.
                # The pool is `rem`.
                # Intersecting cliques are those NOT in disjoint_mask[i].
                # So we keep only those in (disjoint_mask[i] & rem).
                rem = disjoint_mask[i] & rem
            return count

        # Lower Bound: Exact Max Clique on Disjointness Graph (for small subproblems)
        # The Disjointness Graph G_d has edges between disjoint cliques.
        # We want Max Clique in G_d.
        # This is equivalent to Max Independent Set in the Conflict Graph (intersection graph).
        # We implement a simple BnB for Max Clique on the induced subgraph of `remaining_mask`.
        
        def max_clique_exact(remaining_mask, time_limit_inner):
            # Only use if remaining set is small enough
            if remaining_mask.bit_count() > 60:
                return lb_greedy(remaining_mask)
            
            # Extract vertices of the subgraph (indices of active cliques)
            # We need to map global clique indices to local 0..k-1 for the solver
            # Or just use the global indices in the BnB, but that's messy.
            # Let's use a simple recursive BnB that works on the global indices but restricted by mask.
            
            best_size = 0
            t0 = time.time()
            
            # Order cliques by degree in the disjointness graph (within remaining)
            # High degree = more disjoint neighbors = better candidates for clique?
            # Actually, for Max Clique, we usually branch on vertices.
            # Let's use a standard Tomita-like coloring bound if possible, but it's complex with masks.
            # Given the small size (<60), simple BnB with pruning might suffice.
            
            # We will use a greedy coloring bound on the Disjointness Graph induced by remaining_mask.
            # Coloring G_d: If we color G_d with k colors, the size of any clique in G_d is <= k?
            # No. In a graph, Clique Size <= Chromatic Number.
            # So if we color the Disjointness Graph, the max color index used is an UPPER bound on the Max Clique Size.
            # We need a LOWER bound on the Hitting Set, which IS the Max Clique Size.
            # So an Upper Bound on Max Clique Size is NOT a valid Lower Bound.
            # We need a valid Lower Bound.
            # Greedy Disjoint Packing gives a valid Lower Bound.
            # Exact Max Clique gives the Tightest Lower Bound.
            # If we can't compute Exact, we use Greedy.
            
            # Let's try to compute Exact Max Clique for < 60 nodes.
            # Simple BnB:
            def solve_mc(P, X, size):
                nonlocal best_size, t0
                if time.time() - t0 > time_limit_inner:
                    return
                if not P:
                    if size > best_size:
                        best_size = size
                    return
                # Pivot selection: pick v in P|X with max degree in P
                PX = P | X
                if PX:
                    # Find pivot
                    pp = PX
                    best_p = -1
                    best_deg = -1
                    while pp:
                        lb = pp & (-pp)
                        p = lb.bit_length() - 1
                        # Degree in P: count neighbors in P
                        # Neighbors of p in Disjointness Graph are in disjoint_mask[p]
                        deg = (disjoint_mask[p] & P).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_p = p
                        pp ^= lb
                    candidates = P & ~(disjoint_mask[best_p])
                else:
                    candidates = P
                
                cp = candidates
                while cp:
                    lb = cp & (-cp)
                    v = lb.bit_length() - 1
                    cp ^= lb
                    solve_mc(P & disjoint_mask[v], X & disjoint_mask[v], size + 1)
                    P ^= lb
                    X |= lb

            solve_mc(remaining_mask, 0, 0)
            return best_size

        # Branch and Bound for Hitting Set
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        t0_bnb = time.time()
        
        all_clique_mask = (1 << num_cliques) - 1

        def solve_hs(remaining_mask, current_hit, depth):
            nonlocal best_hit, best_size, t0_bnb
            if time.time() - t0_bnb > time_limit:
                return False
            
            if not remaining_mask:
                if len(current_hit) < best_size:
                    best_size = len(current_hit)
                    best_hit = set(current_hit)
                return True
            
            if depth >= best_size:
                return True
            
            # Lower Bound
            # Use exact if small, else greedy
            rem_count = remaining_mask.bit_count()
            if rem_count <= 50:
                lb_val = max_clique_exact(remaining_mask, 0.05) # 50ms limit for LB
            else:
                lb_val = lb_greedy(remaining_mask)
            
            if depth + lb_val >= best_size:
                return True
            
            # Select clique to branch on
            # Heuristic: Smallest clique? Or one with vertices of high frequency?
            # Let's pick the clique that minimizes the branching factor or has rare vertices.
            # Simple: Pick clique with smallest size.
            best_c_i = -1
            best_c_size = n + 1
            rm = remaining_mask
            while rm:
                lb = rm & (-rm)
                i = lb.bit_length() - 1
                sz = clique_masks[i].bit_count()
                if sz < best_c_size:
                    best_c_size = sz
                    best_c_i = i
                rm ^= lb
            
            if best_c_i < 0:
                return True
            
            # Get vertices in best_c_i
            c_mask = clique_masks[best_c_i]
            verts = []
            cm = c_mask
            while cm:
                lb = cm & (-cm)
                v = lb.bit_length() - 1
                verts.append(v)
                cm ^= lb
            
            # Order vertices by descending coverage of remaining cliques
            verts.sort(key=lambda v: -((vertex_clique_mask[v] & remaining_mask).bit_count()))
            
            for v in verts:
                new_hit = current_hit | {v}
                new_remaining = remaining_mask & ~vertex_clique_mask[v]
                if not solve_hs(new_remaining, new_hit, depth + 1):
                    return False
            return True

        solve_hs(all_clique_mask, set(), 0)
        
        # Local Search
        hit = best_hit
        t_ls = time.time()
        improved = True
        while improved and (time.time() - t_ls) < (time_limit * 0.2):
            improved = False
            # Try deletions
            for v in list(hit):
                trial = hit - {v}
                valid = True
                # Check if trial hits all cliques
                # We can use bitmasks for speed
                # For each clique, check if it intersects trial
                # Precomputed: for each clique i, mask of vertices.
                # trial_mask = 0
                # for v in trial: trial_mask |= 1<<v
                # for i in range(num_cliques): if not (clique_masks[i] & trial_mask): invalid
                trial_mask = 0
                for v in trial:
                    trial_mask |= (1 << v)
                
                for i in range(num_cliques):
                    if not (clique_masks[i] & trial_mask):
                        valid = False
                        break
                if valid:
                    hit = trial
                    improved = True
                    break
        
        return sorted(hit)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 5:
            results.append(list(range(n)))
            continue
        
        # Allocate time: 80% to BnB, 20% to Local Search
        bnb_time = remaining * 0.8
        results.append(solve_instance(n, p, seed, bnb_time))

    return results
