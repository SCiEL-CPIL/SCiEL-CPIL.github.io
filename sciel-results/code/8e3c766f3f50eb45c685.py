def dfs(current_size, candidates):
    if current_size >= 10: return True
    if not candidates: return False
    if current_size + len(candidates) < 10: return False
    
    v = candidates[0]
    # Branch 1: Include v
    new_candidates = [u for u in candidates[1:] if u not in adj[v]]
    if dfs(current_size + 1, new_candidates): return True
    
    # Branch 2: Exclude v
    if dfs(current_size, candidates[1:]): return True
    
    return False
