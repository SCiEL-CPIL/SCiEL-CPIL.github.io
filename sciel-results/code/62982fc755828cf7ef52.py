import time
import itertools
import numpy as np

N = 16
DEADLINE = 300.0  # seconds, leave margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # 10s safety margin
    
    points = [(x, y) for x in range(N) for y in range(N)]
    npts = len(points)
    
    # Precompute isosceles conflicts: for each pair (i,j), which k's make isosceles
    # isos[i][j] = set of k such that {i,j,k} is isosceles
    isos = [None] * npts
    for i in range(npts):
        isos[i] = [set() for _ in range(npts)]
    
    for i in range(npts):
        for j in range(i+1, npts):
            for k in range(npts):
                if k != i and k != j:
                    if _is_isosceles(points[i], points[j], points[k]):
                        isos[i][j].add(k)
                        isos[j][i].add(k)
    
    # Greedy minimum maximal independent set
    # S: set of chosen points
    # covered: set of points that are "blocked" (conflict with some pair in S)
    # A point p is blocked if there exist a,b in S such that p is in isos[a][b]
    
    S = []
    S_set = set()
    covered = set()
    
    def is_compatible(p_idx):
        """Check if p_idx can be added to S (no isosceles with any pair in S)."""
        for a in S:
            for b in S:
                if a == b:
                    continue
                if p_idx in isos[a][b]:
                    return False
        return True
    
    def get_new_coverage(p_idx):
        """If we add p_idx to S, which new points get covered?"""
        new_cov = set()
        for a in S:
            # Points that form isosceles with {p_idx, a}
            new_cov |= isos[p_idx][a]
        return new_cov - covered - S_set
    
    while True:
        if time.time() > deadline:
            break
        
        # Check if S is already maximal
        all_points = set(range(npts))
        if all_points - S_set <= covered:
            break
        
        # Find compatible points
        compatible = [i for i in range(npts) if i not in S_set and is_compatible(i)]
        
        if not compatible:
            # Can't add any more points; S is maximal by necessity
            break
        
        # Pick the point that covers the most new points
        best_p = None
        best_cov = -1
        for p in compatible:
            cov = len(get_new_coverage(p))
            if cov > best_cov:
                best_cov = cov
                best_p = p
        
        if best_p is None:
            break
        
        # Add best_p to S
        S.append(best_p)
        S_set.add(best_p)
        # Update covered
        for a in S:
            if a != best_p:
                covered |= isos[best_p][a]
    
    return [points[i] for i in S]
