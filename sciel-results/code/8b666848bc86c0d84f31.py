"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Uses simulated annealing with O(1) edge-delta evaluation.
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
    
    # Build initial dense hexagonal cluster of 37 points, trim to 32
    points = []
    for r in range(4):
        for a in range(-r, r+1):
            for b in range(-r, r+1):
                if (abs(a) + abs(b) + abs(a+b)) // 2 == r:
                    points.append((a,b))
    
    # Trim to 32 by removing lowest-degree points
    pset = set(points)
    while len(pset) > 32:
        min_deg = float('inf')
        min_p = None
        for p in pset:
            deg = sum(1 for n in neighbors(p) if n in pset)
            if deg < min_deg:
                min_deg = deg
                min_p = p
        pset.remove(min_p)
    
    current = list(pset)
    pset = set(current)
    current_edges = count_edges(pset)
    best = list(current)
    best_edges = current_edges
    
    # Simulated annealing
    T = 5.0
    cooling = 0.9995
    max_steps = 100000
    
    for step in range(max_steps):
        i = random.randint(0, 31)
        P = current[i]
        
        # Choose new location Q near P with biased radius
        radius = random.choices([1,2,3,4,5,8,12], weights=[30,25,15,10,8,6,6])[0]
        da = random.randint(-radius, radius)
        db = random.randint(-radius, radius)
        Q = (P[0] + da, P[1] + db)
        
        if Q == P:
            continue
        if abs(Q[0]) > 64 or abs(Q[1]) > 64:
            continue
        if Q in pset:
            continue
        
        # Compute delta in edges (O(1))
        deg_P = sum(1 for n in neighbors(P) if n in pset)
        deg_Q = sum(1 for n in neighbors(Q) if n in pset)
        # If P and Q are adjacent, that edge is lost
        if any(n == Q for n in neighbors(P)):
            deg_Q -= 1
        
        delta = deg_Q - deg_P
        
        if delta > 0 or random.random() < math.exp(delta / T):
            pset.remove(P)
            pset.add(Q)
            current[i] = Q
            current_edges += delta
            
            if current_edges > best_edges:
                best = list(current)
                best_edges = current_edges
        
        T *= cooling
        if T < 0.001:
            break
    
    result = [[a, b] for a, b in best]
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    
    return result
# EVOLVE-BLOCK-END
