def _can_add(p, chosen):
    k = len(chosen)
    if k < 2:
        return True
        
    # Precompute distances from p to all chosen points
    dists_p = [_dist_sq(p, c) for c in chosen]
    
    for i in range(k):
        ai = chosen[i]
        dap = dists_p[i]
        for j in range(i+1, k):
            aj = chosen[j]
            dapj = dists_p[j]
            daij = _dist_sq(ai, aj)
            
            if dap == dapj:
                return False
            if dap == daij:
                return False
            if dapj == daij:
                return False
    return True
