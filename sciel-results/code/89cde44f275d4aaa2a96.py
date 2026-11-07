<<<<<<< SEARCH
    # Phase 2: Greedy growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        current_paths = try_find(current, rng, attempts=100)
        for _ in range(12):
            if time_up():
                break
            if current_paths is None:
                break
            cset = set(current)
            cands = [v for v in range(n) if v not in cset]
            # Score candidates by degree and connectivity to existing branches
            def cand_score(v):
                conn_to_branches = sum(1 for b in current if b in adj[v])
                return (degrees[v], conn_to_branches)
            cands.sort(key=cand_score, reverse=True)
            added = False
            for v in cands[:20]:  # Limit candidates to speed up
                if time_up():
                    break
                new_br = current + [v]
                new_paths = try_find(new_br, rng, attempts=60)
                if new_paths is None:
                    new_paths = try_find_mixed(new_br, rng, attempts=40)
                if new_paths is None:
                    new_paths = try_find_flow(new_br, rng, attempts=20)
                if new_paths is not None:
                    cnt = validate(new_br, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_br, new_paths)
                        current = new_br
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break
=======
    # Phase 2: Greedy growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        current_paths = try_find(current, rng, attempts=100)
        for _ in range(12):
            if time_up():
                break
            if current_paths is None:
                break
            cset = set(current)
            cands = [v for v in range(n) if v not in cset]
            # Score candidates by degree, connectivity to existing branches, and shared neighbors
            def cand_score(v):
                conn_to_branches = sum(1 for b in current if b in adj[v])
                # Count shared neighbors with each existing branch
                shared = sum(len(adj[v] & adj[b]) for b in current)
                return (conn_to_branches, shared, degrees[v])
            cands.sort(key=cand_score, reverse=True)
            added = False
            for v in cands[:20]:  # Limit candidates to speed up
                if time_up():
                    break
                new_br = current + [v]
                new_paths = try_find(new_br, rng, attempts=60)
                if new_paths is None:
                    new_paths = try_find_mixed(new_br, rng, attempts=40)
                if new_paths is None:
                    new_paths = try_find_flow(new_br, rng, attempts=20)
                if new_paths is not None:
                    cnt = validate(new_br, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_br, new_paths)
                        current = new_br
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break
>>>>>>> REPLACE
