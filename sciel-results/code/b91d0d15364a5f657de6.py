"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Uses Analytical Initialization (Hexagon trim) and Greedy Hill Climbing.
    """
    import random
    import math

    random.seed(42)

    def neighbors(p):
        a, b = p
        return [(a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)]

    def count_edges(pset):
        edges = 0
        for p in pset:
            for n in neighbors(p):
                if n in pset:
                    edges += 1
        return edges // 2

    # 1. Analytical Initialization
    # Generate a hexagon of radius 3 (37 points)
    # Points (a,b) such that max(|a|, |b|, |a+b|) <= 3 is not quite right for hex distance.
    # Hex distance from origin for (a,b) is (|a| + |b| + |a+b|) / 2
    points = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            dist = (abs(a) + abs(b) + abs(a+b)) // 2
            if dist <= 3:
                points.append((a, b))
    
    # We have 37 points. We need 32. Remove 5 points with lowest degree.
    pset = set(points)
    
    # Calculate degrees for all points in the 37-point set
    degrees = {}
    for p in pset:
        deg = sum(1 for n in neighbors(p) if n in pset)
        degrees[p] = deg
    
    # Sort points by degree (ascending)
    sorted_points = sorted(pset, key=lambda p: degrees[p])
    
    # Remove the 5 points with the lowest degree
    # If there are ties, we might want to remove those that are 'outer' to keep the core dense, 
    # but removing low degree points is generally safe.
    # To break ties, we can prefer removing points with higher distance from center.
    def point_key(p):
        return (degrees[p], (abs(p[0]) + abs(p[1]) + abs(p[0]+p[1])) // 2)
        
    sorted_points = sorted(pset, key=point_key)
    for p in sorted_points[:5]:
        pset.remove(p)
        
    # Now we have 32 points.
    current = list(pset)
    pset = set(current)
    current_edges = count_edges(pset)
    
    best = list(current)
    best_edges = current_edges

    # 2. Greedy Hill Climbing
    # Try to move each point to a better location
    improved = True
    while improved:
        improved = False
        # Shuffle order to avoid getting stuck in local minima due to fixed ordering
        random.shuffle(current)
        
        for i in range(len(current)):
            P = current[i]
            best_local_P = P
            best_local_edges = current_edges
            
            # Try moving P to nearby empty spots
            # Search radius 2 is usually sufficient for local optima in this context
            for da in range(-2, 3):
                for db in range(-2, 3):
                    Q = (P[0] + da, P[1] + db)
                    if Q == P:
                        continue
                    if abs(Q[0]) > 64 or abs(Q[1]) > 64:
                        continue
                    if Q in pset:
                        continue
                    
                    # Calculate delta
                    deg_P = sum(1 for n in neighbors(P) if n in pset)
                    deg_Q = sum(1 for n in neighbors(Q) if n in pset)
                    
                    # If P and Q are adjacent, that edge is lost in the move (it becomes a self-loop or is removed)
                    # Actually, if we move P to Q, the edge (P, X) is removed for all X in N(P).
                    # The edge (Q, X) is added for all X in N(Q).
                    # If X is both in N(P) and N(Q), the edge count change is 0 for that X.
                    # The formula: delta = deg_Q - deg_P - (1 if P in N(Q) else 0) ?
                    # Let's be precise.
                    # Edges removed: all edges (P, x) where x in N(P) and x in pset. Count = deg_P.
                    # Edges added: all edges (Q, x) where x in N(Q) and x in pset.
                    # Note: P is in pset. If P is in N(Q), then the edge (Q, P) is added.
                    # But wait, we are moving P to Q. So P is no longer at P.
                    # The set of other points is pset \ {P}.
                    # Let S = pset \ {P}.
                    # Edges removed: (P, x) for x in N(P) intersect S.
                    # Edges added: (Q, x) for x in N(Q) intersect S.
                    
                    S = pset - {P}
                    removed = sum(1 for n in neighbors(P) if n in S)
                    added = sum(1 for n in neighbors(Q) if n in S)
                    
                    delta = added - removed
                    
                    if delta > 0:
                        if current_edges + delta > best_local_edges:
                            best_local_edges = current_edges + delta
                            best_local_P = Q
            
            if best_local_P != P:
                # Perform the move
                pset.remove(P)
                pset.add(best_local_P)
                current[i] = best_local_P
                current_edges = best_local_edges
                improved = True
                
                if current_edges > best_edges:
                    best = list(current)
                    best_edges = current_edges

    result = [[a, b] for a, b in best]
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    
    return result
# EVOLVE-BLOCK-END
