"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64

# EVOLVE-BLOCK-START
def construct():
    def greedy(points_order):
        points = []
        dist_sets = []
        
        for Q in points_order:
            qx, qy = Q
            can_add = True
            new_dists = []
            
            for i, P in enumerate(points):
                px, py = P
                dx = qx - px
                dy = qy - py
                d2 = dx * dx + dy * dy
                new_dists.append(d2)
                if d2 in dist_sets[i]:
                    can_add = False
                    break
            
            if can_add:
                if len(set(new_dists)) == len(new_dists):
                    points.append(Q)
                    dist_sets.append(set(new_dists))
        
        return points
    
    # Try several orderings and return the best
    orderings = []
    
    # Diagonal sweep: (x+y, x)
    diag = []
    for s in range(2 * N - 1):
        for x in range(max(0, s - N + 1), min(N, s + 1)):
            y = s - x
            if 0 <= y < N:
                diag.append((x, y))
    orderings.append(diag)
    
    # Row-major
    row_major = [(x, y) for y in range(N) for x in range(N)]
    orderings.append(row_major)
    
    # Column-major
    col_major = [(x, y) for x in range(N) for y in range(N)]
    orderings.append(col_major)
    
    # Anti-diagonal: (x-y+N, x)
    anti = []
    for d in range(-N + 1, N):
        for x in range(max(0, -d), min(N, N + d)):
            y = x - d
            if 0 <= y < N:
                anti.append((x, y))
    orderings.append(anti)
    
    best = []
    for order in orderings:
        result = greedy(order)
        if len(result) > len(best):
            best = result
    
    return best
# EVOLVE-BLOCK-END
