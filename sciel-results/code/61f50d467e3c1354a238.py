import random
import time
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Precompute all pair distances? No, dynamic is better for memory if S is small, 
    # but we can precompute a lookup table for all 256x256 distances.
    # 256*256 = 65536 ints. Very small.
    dist_table = [[0]*N for _ in range(N)] # No, points are (x,y).
    # Let's use a dict or 2D list indexed by x,y.
    # dist[x1][y1][x2][y2] is too big if nested.
    # Just compute on fly or cache in a dict.
    # With 256 points, dict lookup is fast enough, or just compute.
    # Computing (dx*dx + dy*dy) is ~5 ops.
    
    def try_greedy(order):
        S = []
        # dist_from[i] is a set of squared distances from S[i] to other points in S
        dist_from = []
        
        for p in order:
            # Check if p can be added
            valid = True
            
            # 1. Check if p forms isosceles with vertex p: d(p,a) == d(p,b)
            # Compute distances from p to all a in S
            dp = []
            for a in S:
                d = _d2(p, a)
                dp.append(d)
            
            if len(set(dp)) != len(dp):
                valid = False
            
            if valid:
                # 2. Check if p forms isosceles with vertex a or b
                # For each a in S, check if d(p,a) is in dist_from[a]
                for i, a in enumerate(S):
                    if dp[i] in dist_from[i]:
                        valid = False
                        break
            
            if valid:
                # Add p
                S.append(p)
                new_dist_set = set()
                for i, a in enumerate(S[:-1]):
                    d = dp[i]
                    # Add d to dist_from[i]
                    dist_from[i].add(d)
                    # Add d to new_dist_set for p
                    new_dist_set.add(d)
                dist_from.append(new_dist_set)
                
        return S

    # Run multiple random restarts
    num_restarts = 0
    while time.time() < deadline:
        random.shuffle(POINTS)
        # Use a copy to shuffle
        order = POINTS[:]
        random.shuffle(order)
        
        res = try_greedy(order)
        if len(res) < best_size:
            best_size = len(res)
            best_set = res
            # Early stop if we find a very small set?
            # Theoretical min is likely around 10-15.
            if best_size <= 10:
                break
        
        num_restarts += 1
        
    if best_set is None:
        # Fallback: simple greedy
        best_set = try_greedy(POINTS)
        
    return best_set
