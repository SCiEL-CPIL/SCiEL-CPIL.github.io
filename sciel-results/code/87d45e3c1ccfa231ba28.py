    # Backtracking with limit
    visit_count = 0
    max_visits = 100000
    
    def backtrack(current_clique, candidates):
        nonlocal visit_count
        visit_count += 1
        if visit_count > max_visits:
            return None
        if time.time() - start > time_limit:
            return None
        if len(current_clique) == n:
            return current_clique
        if len(current_clique) + len(candidates) < n:
            return None
            
        # Order candidates by degree to find solution faster
        # Sorting at every step is expensive. 
        # We can just iterate.
        for v in candidates:
            new_candidates = candidates.intersection(adj[v])
            if len(current_clique) + 1 + len(new_candidates) >= n:
                res = backtrack(current_clique + [v], new_candidates)
                if res:
                    return res
        return None
