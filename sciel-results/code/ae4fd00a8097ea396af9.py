<<<<<<< SEARCH
    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # N[idx] = bitmask of C-vertices adjacent to I-vertex (idx + k)
        N = [0] * (n - k)
        for u, v in edges:
            if u < k and v >= k:
                N[v - k] |= (1 << u)
            elif v < k and u >= k:
                N[u - k] |= (1 << v)

        # rem[u] = bitmask of vertices v where C-edge (u,v) is not yet covered
        # Initially all pairs in C are uncovered
        rem = [0] * k
        for u in range(k):
            rem[u] = ((1 << k) - 1) ^ (1 << u)

        cliques = []

        # Phase 1: Process I-vertices by decreasing neighborhood size
        I_order = sorted(range(n - k), key=lambda idx: bin(N[idx]).count('1'), reverse=True)

        for idx in I_order:
            if time.time() > deadline:
                break
            i = idx + k
            mask = N[idx]
            if mask == 0:
                continue

            while mask:
                if time.time() > deadline:
                    break

                # Greedy max clique in subgraph induced by mask (using uncovered C-edges)
                S = []
                candidates = mask
                while candidates:
                    if time.time() > deadline:
                        break
                    best_v = -1
                    best_deg = -1
                    cm = candidates
                    while cm:
                        lsb = cm & (-cm)
                        v = lsb.bit_length() - 1
                        deg = bin(rem[v] & candidates).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                        cm ^= lsb
                    if best_v < 0:
                        break
                    S.append(best_v)
                    candidates &= rem[best_v]
                    candidates &= ~(1 << best_v)

                if not S:
                    lsb = mask & (-mask)
                    S = [lsb.bit_length() - 1]

                clique = sorted([i] + S)
                cliques.append(clique)

                # Mark edges within S as covered
                for a_idx in range(len(S)):
                    for b_idx in range(a_idx + 1, len(S)):
                        a, b = S[a_idx], S[b_idx]
                        rem[a] &= ~(1 << b)
                        rem[b] &= ~(1 << a)

                for v in S:
                    mask &= ~(1 << v)

        # Phase 2: Cover remaining C-edges with C-only cliques
        active = set()
        for u in range(k):
            if rem[u]:
                active.add(u)

        while active:
            if time.time() > deadline:
                break

            # Pick vertex with max degree in active subgraph
            best_v = -1
            best_deg = -1
            for v in active:
                cnt = 0
                for a in active:
                    if a != v and (rem[v] >> a) & 1:
                        cnt += 1
                if cnt > best_deg:
                    best_deg = cnt
                    best_v = v

            if best_v < 0 or best_deg == 0:
                break

            # Greedy clique expansion
            S = [best_v]
            candidates = rem[best_v]
            active_mask = 0
            for a in active:
                active_mask |= (1 << a)
            candidates &= active_mask
            candidates &= ~(1 << best_v)

            while candidates:
                if time.time() > deadline:
                    break
                best_w = -1
                best_wdeg = -1
                cm = candidates
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    deg = bin(rem[w] & candidates).count('1')
                    if deg > best_wdeg:
                        best_wdeg = deg
                        best_w = w
                    cm ^= lsb
                if best_w < 0:
                    break
                S.append(best_w)
                candidates &= rem[best_w]
                candidates &= ~(1 << best_w)

            if len(S) < 2:
                nbrs = [a for a in active if a != best_v and (rem[best_v] >> a) & 1]
                if nbrs:
                    S = [best_v, nbrs[0]]
                else:
                    active.discard(best_v)
                    continue

            S_list = sorted(S)
            cliques.append(S_list)

            for a_idx in range(len(S_list)):
                for b_idx in range(a_idx + 1, len(S_list)):
                    a, b = S_list[a_idx], S_list[b_idx]
                    rem[a] &= ~(1 << b)
                    rem[b] &= ~(1 << a)

            active = set(v for v in active if rem[v])

        return cliques
=======
    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # N[idx] = bitmask of C-vertices adjacent to I-vertex (idx + k)
        N = [0] * (n - k)
        for u, v in edges:
            if u < k and v >= k:
                N[v - k] |= (1 << u)
            elif v < k and u >= k:
                N[u - k] |= (1 << v)

        # rem[u] = bitmask of vertices v where C-edge (u,v) is not yet covered
        rem = [0] * k
        all_c_mask = (1 << k) - 1
        for u in range(k):
            rem[u] = all_c_mask ^ (1 << u)

        # cross_rem[i] = bitmask of C-neighbors of I-vertex i not yet covered by any clique
        cross_rem = list(N)

        cliques = []
        nI = n - k

        def expand_clique(start_mask, rem_arr, max_size=8):
            """Greedy clique expansion within start_mask using rem_arr as edge filter."""
            S = []
            candidates = start_mask
            while candidates and len(S) < max_size:
                best_v = -1
                best_deg = -1
                cm = candidates
                while cm:
                    lsb = cm & (-cm)
                    v = lsb.bit_length() - 1
                    deg = bin(rem_arr[v] & candidates).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cm ^= lsb
                if best_v < 0:
                    break
                S.append(best_v)
                candidates &= rem_arr[best_v]
                candidates &= ~(1 << best_v)
            return S

        while True:
            if time.time() > deadline:
                break

            # Check if all edges are covered
            any_cross = False
            for idx in range(nI):
                if cross_rem[idx]:
                    any_cross = True
                    break
            if not any_cross:
                any_c = False
                for u in range(k):
                    if rem[u]:
                        any_c = True
                        break
                if not any_c:
                    break

            # Global greedy: find the candidate clique covering most uncovered edges
            best_gain = -1
            best_clique = None
            best_type = None  # 'cross' or 'c_only'
            best_i = -1
            best_S = []

            # Evaluate cross-cliques: for each I-vertex with remaining cross edges
            for idx in range(nI):
                if time.time() > deadline:
                    break
                cm = cross_rem[idx]
                if not cm:
                    continue
                # Greedy expand within cm using rem (C-edges among S)
                S = expand_clique(cm, rem)
                if not S:
                    lsb = cm & (-cm)
                    S = [lsb.bit_length() - 1]
                # Gain = uncovered cross edges covered + uncovered C-edges within S
                gain = 0
                for v in S:
                    gain += bin(rem[v] & cm).count('1')  # C-edges within S that are uncovered
                # Cross edges covered: all edges from i to S
                gain += len(S)
                if gain > best_gain:
                    best_gain = gain
                    best_clique = S
                    best_type = 'cross'
                    best_i = idx

            # Evaluate C-only cliques: pick a few starting vertices
            # Use top-degree vertices as seeds
            c_seeds = []
            for u in range(k):
                if rem[u]:
                    deg = bin(rem[u]).count('1')
                    c_seeds.append((deg, u))
            c_seeds.sort(reverse=True)
            for deg, u in c_seeds[:15]:
                if time.time() > deadline:
                    break
                S = expand_clique(rem[u], rem)
                if len(S) < 2:
                    continue
                # Gain = C-edges within S that are uncovered
                gain = 0
                for a_idx in range(len(S)):
                    for b_idx in range(a_idx + 1, len(S)):
                        a, b = S[a_idx], S[b_idx]
                        if (rem[a] >> b) & 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_clique = S
                    best_type = 'c_only'
                    best_i = -1

            if best_gain <= 0:
                # Fallback: just cover one edge
                # Try cross first
                found = False
                for idx in range(nI):
                    if cross_rem[idx]:
                        lsb = cross_rem[idx] & (-cross_rem[idx])
                        v = lsb.bit_length() - 1
                        # Need at least 2 vertices: find another C-neighbor or use C-edge
                        # Use {i, v} if there's another C-vertex connected to v via rem, else pair with any C
                        S = [v]
                        # Try to expand
                        S = expand_clique(cross_rem[idx], rem)
                        if not S:
                            S = [v]
                        if len(S) < 2:
                            # Pair v with any other C-vertex that has an edge to v in rem
                            other = rem[v] & cross_rem[idx]
                            if other:
                                lsb2 = other & (-other)
                                S.append(lsb2.bit_length() - 1)
                            else:
                                # Just use {i, v} - but that's only covering cross edge, need clique of size>=2
                                # {i, v} is a clique since i is adjacent to v
                                S = [v]
                        cliques.append(sorted([idx + k] + S))
                        for a_idx in range(len(S)):
                            for b_idx in range(a_idx + 1, len(S)):
                                a, b = S[a_idx], S[b_idx]
                                rem[a] &= ~(1 << b)
                                rem[b] &= ~(1 << a)
                        for v2 in S:
                            cross_rem[idx] &= ~(1 << v2)
                        found = True
                        break
                if not found:
                    # Cover a C-edge
                    for u in range(k):
                        if rem[u]:
                            lsb = rem[u] & (-rem[u])
                            v = lsb.bit_length() - 1
                            cliques.append(sorted([u, v]))
                            rem[u] &= ~(1 << v)
                            rem[v] &= ~(1 << u)
                            break
                    if not any(rem[u] for u in range(k)):
                        break
                continue

            # Apply the best clique
            if best_type == 'cross':
                S = best_clique
                i = best_i
                clique = sorted([i + k] + S)
                cliques.append(clique)
                # Update rem: mark C-edges within S as covered
                for a_idx in range(len(S)):
                    for b_idx in range(a_idx + 1, len(S)):
                        a, b = S[a_idx], S[b_idx]
                        rem[a] &= ~(1 << b)
                        rem[b] &= ~(1 << a)
                # Update cross_rem: mark cross edges from i to S as covered
                for v in S:
                    cross_rem[i] &= ~(1 << v)
            else:
                S = best_clique
                S_sorted = sorted(S)
                cliques.append(S_sorted)
                for a_idx in range(len(S_sorted)):
                    for b_idx in range(a_idx + 1, len(S_sorted)):
                        a, b = S_sorted[a_idx], S_sorted[b_idx]
                        rem[a] &= ~(1 << b)
                        rem[b] &= ~(1 << a)

        return cliques
>>>>>>> REPLACE

