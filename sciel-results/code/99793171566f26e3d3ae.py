import time
import random
import math

N = 64
POINTS = [(x, y) for x in range(N) for y in range(N)]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    
    # Pre-compute squares for fast distance checks? Not strictly needed if we solve equations.
    
    while time.time() < deadline:
        S = []
        available = set(POINTS)
        
        # Greedy construction
        while available:
            # Pick a random point
            p = random.choice(list(available))
            S.append(p)
            available.remove(p)
            
            # Update available by removing points that form isosceles with p and any existing q in S
            # Note: We only need to check against previously added points
            # Because if p, q, r was isosceles, it would have been caught when r was added (if q was already there)
            # Or when q was added (if r was already there).
            # Since we add one by one, we just need to check p against all existing q in S[:-1].
            
            new_forbidden = set()
            for q in S[:-1]:
                new_forbidden.update(find_isosceles_partners(p, q))
            
            available -= new_forbidden
        
        if len(S) > len(best_set):
            best_set = S[:]
            
            # Optional: Try local search to improve
            # If we have time, we can try to swap points
            # But simple greedy with many restarts is usually very strong for this size.
            
    return best_set

def find_isosceles_partners(p, q):
    """
    Returns a set of points r such that p, q, r form an isosceles triangle.
    p, q are (x, y) tuples.
    """
    x1, y1 = p
    x2, y2 = q
    
    # Case 1: d(p, r) == d(p, q)
    # (x - x1)^2 + (y - y1)^2 = (x1 - x2)^2 + (y1 - y2)^2
    dx12 = x1 - x2
    dy12 = y1 - y2
    d2 = dx12*dx12 + dy12*dy12
    
    forbidden = set()
    
    # Circle centered at p
    for x in range(N):
        dx = x - x1
        dy2 = d2 - dx*dx
        if dy2 < 0:
            continue
        dy = int(math.isqrt(dy2))
        if dy*dy == dy2:
            y = y1 + dy
            if 0 <= y < N:
                forbidden.add((x, y))
            y = y1 - dy
            if 0 <= y < N:
                forbidden.add((x, y))
                
    # Circle centered at q
    for x in range(N):
        dx = x - x2
        dy2 = d2 - dx*dx
        if dy2 < 0:
            continue
        dy = int(math.isqrt(dy2))
        if dy*dy == dy2:
            y = y2 + dy
            if 0 <= y < N:
                forbidden.add((x, y))
            y = y2 - dy
            if 0 <= y < N:
                forbidden.add((x, y))
                
    # Perpendicular bisector of pq
    # (x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2
    # 2(x2 - x1)x + 2(y2 - y1)y = x2^2 - x1^2 + y2^2 - y1^2
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = x2*x2 - x1*x1 + y2*y2 - y1*y1
    
    if A == 0 and B == 0:
        # p == q, not possible in set
        pass
    elif A == 0:
        # By = C
        if C % B == 0:
            y = C // B
            if 0 <= y < N:
                for x in range(N):
                    forbidden.add((x, y))
    elif B == 0:
        # Ax = C
        if C % A == 0:
            x = C // A
            if 0 <= x < N:
                for y in range(N):
                    forbidden.add((x, y))
    else:
        # Ax + By = C
        # Iterate x
        for x in range(N):
            rem = C - A * x
            if rem % B == 0:
                y = rem // B
                if 0 <= y < N:
                    forbidden.add((x, y))
                    
    return forbidden
