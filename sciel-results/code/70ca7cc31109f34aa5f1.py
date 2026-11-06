    # Strategy: Randomized Greedy with Perturbation
    # This is more effective than the complex branch-and-bound for n=16
    while time.time() < deadline - 10.0:
        # Start from best found or base
        adj = [a for a in best_adj] if best_cnt > base_cnt else [a for a in base_adj]
        
        # Perturb
        existing = _get_existing_edges(adj)
        num_remove = rng.randint(0, 3)
        for _ in range(num_remove):
            if existing:
                u, v = rng.choice(existing)
                _remove_edge(adj, u, v)
                existing.remove((u, v))
                
        # Greedy fill
        missing = _get_missing_edges(adj)
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 10.0:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                if _is_not_3_colorable(adj):
                    cnt = _edge_count(adj)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_adj = [a for a in adj]
                else:
                    _remove_edge(adj, u, v)
