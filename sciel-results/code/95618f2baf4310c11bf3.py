"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    # If any two distances are equal, it's isosceles
    # Also handle degenerate case (collinear) - if d1==0 or d2==0 or d3==0
    # Actually, for grid points, we can have equal distances
    return d1 == d2 or d1 == d3 or d2 == d3


def construct():
    """
    Find a minimal maximal no-isosceles set in [0, N)^2.
    
    Strategy: 
    1. Start with an empty set.
    2. Use a greedy approach with randomized ordering to find a maximal set.
    3. Then try to remove points while maintaining maximality to minimize size.
    
    Actually, a better approach:
    1. First, find a maximal no-isosceles set (any size).
    2. Then, iteratively try to remove points that are not "essential" for maximality.
    3. A point p in S is essential if removing it allows adding some point q.
       So p is non-essential if for every q not in S, q conflicts with some other point in S\{p}.
    
    Let me think about this more carefully:
    - S is maximal no-isosceles if no point can be added.
    - S is minimal maximal if no proper subset of S is also maximal.
    - A point p is essential for maximality if S \ {p} is NOT maximal, i.e., there exists some q not in S\{p} such that (S\{p}) ∪ {q} is no-isosceles.
    - So p is non-essential if for all q not in S, (S\{p}) ∪ {q} is NOT no-isosceles, meaning q conflicts with some point in S\{p}.
    
    Algorithm:
    1. Generate a maximal no-isosceles set using greedy (randomized).
    2. For each point p in S (in some order), check if p is essential.
       If not essential, remove it.
    3. Repeat until no more points can be removed.
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Step 1: Greedy construction with randomization
    import random
    random.seed(42)
    
    # Try multiple random orderings and pick the best
    best_set = None
    best_size = float('inf')
    
    for trial in range(5):
        random.shuffle(points)
        chosen = []
        for p in points:
            can_add = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if _is_isosceles(p, chosen[i], chosen[j]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen.append(p)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    if best_set is None:
        return []
    
    S = set(best_set)
    
    # Step 2: Remove non-essential points
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias
        pts = list(S)
        random.shuffle(pts)
        for p in pts:
            if p not in S:
                continue
            # Check if p is essential: is there a q not in S such that 
            # (S - {p}) ∪ {q} is no-isosceles?
            # If no such q exists, p is non-essential, remove it.
            is_essential = False
            for q in points:
                if q in S:
                    continue
                # Check if (S - {p}) ∪ {q} is no-isosceles
                valid = True
                for a, b in itertools.combinations(S - {p}, 2):
                    if _is_isosceles(q, a, b):
                        valid = False
                        break
                if valid:
                    # q can be added to S\{p}, so p is essential
                    is_essential = True
                    break
            
            if not is_essential:
                S.remove(p)
                changed = True
    
    return list(S)
# EVOLVE-BLOCK-END
