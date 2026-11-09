import math

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def construct():
    n = 32
    # Use hexagonal spiral construction which is known to be optimal
    # for this problem size (attains Harborth's bound).
    R = 3
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                hd = _hexdist(a,b)
                x = a + 0.5*b
                y = 0.8660254037844386*b
                ang = math.atan2(y, x)
                cands.append((hd, ang, a, b))
                
    cands.sort()
    # Take the first n points
    best = [(a,b) for hd, ang, a, b in cands[:n]]
    
    return [[a,b] for (a,b) in best]
