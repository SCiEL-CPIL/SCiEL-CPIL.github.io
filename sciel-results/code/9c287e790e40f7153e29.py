<<<<<<< SEARCH
    # ------------------------------------------------------------------
    # Joint value+edge local search (recompute score on value moves,
    # incremental on edge moves).
    # ------------------------------------------------------------------
    def joint_search(vals0, edges0, steps, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        no_improve = 0
        consecutive_no_improve = 0

        for _ in range(steps):
            if time.time() > deadline:
                break

            if random.random() < 0.5:
                # edge move (incremental) - try multiple candidates
                i = random.randrange(m)
                e_out = edges[i]

                best_e_in = None
                best_r = float('inf')
                best_swap_data = None

                for _ in range(3):
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue

                    swap(e_out, e_in)
                    r = raw()

                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                        best_swap_data = (e_out, e_in)

                    swap(e_in, e_out)

                if best_e_in is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue

                e_out, e_in = best_swap_data
                swap(e_out, e_in)
                r = raw()

                if r <= cur:
                    cur = r
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    swap(e_in, e_out)
                    no_improve += 1
                    consecutive_no_improve += 1
            else:
                # value move (recompute) - smarter proposals
                idx = random.randrange(n)
                old = vals[idx]

                # Generate multiple candidate values and pick the best
                candidates = []

                # Small values
                for _ in range(3):
                    candidates.append(random.randint(1, 30))
                # Medium values
                for _ in range(2):
                    candidates.append(random.randint(10, 60))
                # Related to existing values
                for _ in range(2):
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        candidates.append(vals[other_idx] + random.randint(-5, 5))
                        candidates.append(vals[other_idx] * random.choice([2, 3]))

                best_nv = None
                best_r = float('inf')

                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue

                    vals[idx] = nv
                    r = full_score(vals, edges)

                    if r < best_r:
                        best_r = r
                        best_nv = nv

                    vals[idx] = old

                if best_nv is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue

                vals[idx] = best_nv
                r = best_r

                if r <= cur:
                    cur = r
                    # rebuild incremental state (only on accepted value move)
                    init_from, swap, add_edge, raw, _ = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1
                    consecutive_no_improve += 1

        return best, best_vals, best_edges
=======
    # ------------------------------------------------------------------
    # Joint value+edge local search (recompute score on value moves,
    # incremental on edge moves).
    # ------------------------------------------------------------------
    def joint_search(vals0, edges0, steps, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, raw_smooth, _ = make_scoring(vals)
        init_from(edges)
        no_improve = 0
        consecutive_no_improve = 0

        # Precompute edge sums/products for biased edge selection
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            edge_sum[(u, v)] = vals[u] + vals[v]
            edge_prod[(u, v)] = vals[u] * vals[v]

        sum_to_edges = {}
        prod_to_edges = {}
        for u, v in all_edges:
            s = edge_sum[(u, v)]
            p = edge_prod[(u, v)]
            sum_to_edges.setdefault(s, []).append((u, v))
            prod_to_edges.setdefault(p, []).append((u, v))

        for _ in range(steps):
            if time.time() > deadline:
                break

            if random.random() < 0.5:
                # edge move (incremental) - try multiple candidates
                i = random.randrange(m)
                e_out = edges[i]
                s_out = edge_sum[e_out]
                p_out = edge_prod[e_out]

                best_e_in = None
                best_r = float('inf')
                best_swap_data = None

                for _ in range(6):
                    if random.random() < 0.4:
                        candidates_same_sum = sum_to_edges.get(s_out, [])
                        if candidates_same_sum:
                            e_in = random.choice(candidates_same_sum)
                        else:
                            e_in = random.choice(all_edges)
                    elif random.random() < 0.7:
                        candidates_same_prod = prod_to_edges.get(p_out, [])
                        if candidates_same_prod:
                            e_in = random.choice(candidates_same_prod)
                        else:
                            e_in = random.choice(all_edges)
                    else:
                        e_in = random.choice(all_edges)

                    if e_in in edges_set:
                        continue

                    swap(e_out, e_in)
                    r = raw()

                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                        best_swap_data = (e_out, e_in)

                    swap(e_in, e_out)

                if best_e_in is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue

                e_out, e_in = best_swap_data
                swap(e_out, e_in)
                r = raw()

                if r <= cur:
                    cur = r
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    # Update edge_sum/edge_prod for the new edge
                    edge_sum[e_in] = vals[e_in[0]] + vals[e_in[1]]
                    edge_prod[e_in] = vals[e_in[0]] * vals[e_in[1]]
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    swap(e_in, e_out)
                    no_improve += 1
                    consecutive_no_improve += 1
            else:
                # value move (recompute) - smarter proposals
                idx = random.randrange(n)
                old = vals[idx]

                # Generate multiple candidate values and pick the best
                candidates = []

                # Small values
                for _ in range(4):
                    candidates.append(random.randint(1, 25))
                # Medium values
                for _ in range(3):
                    candidates.append(random.randint(5, 50))
                # Related to existing values
                for _ in range(3):
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        candidates.append(vals[other_idx] + random.randint(-3, 3))
                        candidates.append(vals[other_idx] * random.choice([2, 3, 4]))
                        # Try to create sum collisions
                        target_sum = random.choice([vals[e[0]] + vals[e[1]] for e in edges])
                        candidates.append(target_sum - vals[other_idx])
                # Try values that are multiples of small numbers
                for mult in [2, 3, 4, 5, 6]:
                    candidates.append(random.randint(1, 20) * mult)

                best_nv = None
                best_r = float('inf')

                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue

                    vals[idx] = nv
                    r = full_score(vals, edges)

                    if r < best_r:
                        best_r = r
                        best_nv = nv

                    vals[idx] = old

                if best_nv is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue

                vals[idx] = best_nv
                r = best_r

                if r <= cur:
                    cur = r
                    # rebuild incremental state (only on accepted value move)
                    init_from, swap, add_edge, raw, raw_smooth, _ = make_scoring(vals)
                    init_from(edges)
                    # Update edge_sum/edge_prod
                    for u, v in all_edges:
                        edge_sum[(u, v)] = vals[u] + vals[v]
                        edge_prod[(u, v)] = vals[u] * vals[v]
                    sum_to_edges = {}
                    prod_to_edges = {}
                    for u, v in all_edges:
                        s = edge_sum[(u, v)]
                        p = edge_prod[(u, v)]
                        sum_to_edges.setdefault(s, []).append((u, v))
                        prod_to_edges.setdefault(p, []).append((u, v))
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1
                    consecutive_no_improve += 1

        return best, best_vals, best_edges
>>>>>>> REPLACE

