<<<<<<< SEARCH
    # Phase 5: Intensive local improvement on best
    if time.time() - t0 < DEADLINE - 5:
        adj = _adjacency(best)
        deg = _degrees(adj)
        for _ in range(200):
            if time.time() - t0 > DEADLINE - 2:
                break
            if not _fast_is_valid(adj, deg):
                break
            # Try to add a random missing edge
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj[u] >> v) & 1]
            if not missing:
                break
            u, v = rng.choice(missing)
            trial_adj = adj[:]
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_deg = deg[:]
            trial_deg[u] += 1
            trial_deg[v] += 1
            if _fast_is_valid(trial_adj, trial_deg):
                adj = trial_adj
                deg = trial_deg
                best = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
                best_len = len(best)
=======
    # Phase 5: Intensive local improvement on best
    if time.time() - t0 < DEADLINE - 5:
        adj = _adjacency(best)
        deg = _degrees(adj)
        # Try all missing edges in multiple passes
        for _ in range(10):
            if time.time() - t0 > DEADLINE - 2:
                break
            improved = False
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj[u] >> v) & 1]
            for u, v in missing:
                if time.time() - t0 > DEADLINE - 2:
                    break
                trial_adj = adj[:]
                trial_adj[u] |= (1 << v)
                trial_adj[v] |= (1 << u)
                trial_deg = deg[:]
                trial_deg[u] += 1
                trial_deg[v] += 1
                if _fast_is_valid(trial_adj, trial_deg):
                    adj = trial_adj
                    deg = trial_deg
                    improved = True
            if not improved:
                break
        best = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
        best_len = len(best)
>>>>>>> REPLACE
