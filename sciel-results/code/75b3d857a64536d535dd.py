import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)
    
    # Precompute field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Addition: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Multiplication: (a1+b1*alpha)*(a2+b2*alpha) = (a1*a2 + b1*b2*k) + (a1*b2 + a2*b1)*alpha
    
    def add_el(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def sub_el(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
    
    def mul_el(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1*a2 + b1*b2*k) % p, (a1*b2 + a2*b1) % p)
    
    def to_idx(e):
        return e[0] * p + e[1]
    
    def from_idx(idx):
        return (idx // p, idx % p)
    
    # Total points: 49*49 = 2401
    total = p * p * p * p  # xa, xb, ya, yb each 0..6
    
    # We want to find the largest set S such that no line contains more than one point of S.
    # Lines are of form {x + t*v : t in F_49} where v = (1, m) for m in F_49, or v = (0, 1).
    
    # For each direction, we need to ensure at most one point of S lies on any line in that direction.
    # Direction v = (0, 1): lines are vertical, x = constant. So for each x, at most one y in S with that x.
    # Direction v = (1, m): lines are y = m*(x - x0) + y0. 
    
    # Greedy approach: try to add points to S in an order that maximizes |S|.
    # A point can be added to S if it doesn't conflict with existing points in S on any line.
    
    # For efficiency, track for each direction and each line, how many points of S are on it.
    
    # Let's use a different approach: since we want to maximize |S|, and the constraint is that
    # no line has >1 point of S, this is equivalent to finding a maximum independent set in a
    # conflict graph where two points conflict if they're on the same line.
    
    # But that's NP-hard. Let's use a greedy heuristic.
    
    # Key observation: for direction (0,1), each vertical line (fixed x) can have at most 1 point in S.
    # There are 49 vertical lines, so |S| <= 49 from this constraint alone.
    # Similarly, for each direction (1, m), there are 49 lines, each with at most 1 point in S.
    
    # Actually, the maximum size of S is bounded by the minimum over all directions of the number of lines
    # in that direction, which is 49 for all directions. So |S| <= 49.
    
    # Can we achieve |S| = 49? That would be a set of 49 points, one on each vertical line,
    # and also one on each line of every other direction. This is like a transversal.
    
    # Let's try: pick one point on each vertical line (x = constant) such that for every other direction,
    # no two points are on the same line.
    
    # For direction (1, m), the line through (x0, y0) has equation y = m*(x - x0) + y0.
    # Two points (x1, y1) and (x2, y2) are on the same line of direction (1,m) iff
    # y1 - y2 = m*(x1 - x2).
    
    # So if we pick a function f: F_49 -> F_49 (choosing y = f(x) for each x), then
    # two points (x1, f(x1)) and (x2, f(x2)) are on the same line of direction (1,m) iff
    # f(x1) - f(x2) = m*(x1 - x2).
    
    # We need this to never happen for any m in F_49 (except when x1 = x2, which is the same point).
    
    # This is equivalent to saying that the function f has the property that for any distinct x1, x2,
    # the ratio (f(x1) - f(x2))/(x1 - x2) is never in F_49... wait, that's always in F_49.
    
    # Actually, we need that for any distinct x1, x2, and any m, f(x1) - f(x2) != m*(x1 - x2).
    # But m ranges over all of F_49, and (x1-x2) is nonzero, so m*(x1-x2) ranges over all of F_49.
    # So f(x1) - f(x2) is always equal to m*(x1-x2) for some m (namely m = (f(x1)-f(x2))/(x1-x2)).
    
    # This means it's IMPOSSIBLE to have 49 points with one on each vertical line and no two on the same
    # line of any other direction! Because any two points determine a unique direction.
    
    # So |S| < 49. Let's think more carefully.
    
    # The constraint is: for each direction, at most one point of S on each line in that direction.
    # For direction (0,1): 49 lines, at most 1 point each, so |S| <= 49.
    # For direction (1, m) for a specific m: 49 lines, at most 1 point each, so |S| <= 49.
    
    # But the constraints interact. Let's just do a greedy search.
    
    start_time = time.time()
    deadline = 600  # seconds
    
    # Represent points as indices 0..2400
    # Point index = xa*343 + xb*49 + ya*7 + yb
    
    def idx_to_point(idx):
        xa = idx // 343
        xb = (idx // 49) % 7
        ya = (idx // 7) % 7
        yb = idx % 7
        return (xa, xb, ya, yb)
    
    def point_to_idx(xa, xb, ya, yb):
        return xa*343 + xb*49 + ya*7 + yb
    
    # For greedy, we need to check if adding a point conflicts with existing points in S.
    # Two points conflict if they are on the same line in some direction.
    # Directions: (0,1) and (1, m) for m in F_49.
    
    # For direction (0,1): same x coordinate.
    # For direction (1, m): (x2-x1) and (y2-y1) satisfy y2-y1 = m*(x2-x1).
    
    # Precompute for each pair of points whether they conflict? That's 2401^2/2 ~ 2.8M pairs, feasible.
    
    # Actually, let's just do a greedy with a priority: try to add points that don't conflict.
    
    # Better: use a randomized greedy multiple times and keep the best.
    
    best_S = set()
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() - start_time > deadline - 30:
            break
        
        S = set()
        # Shuffle points
        points_list = list(range(total))
        np.random.shuffle(points_list)
        
        for idx in points_list:
            xa, xb, ya, yb = idx_to_point(idx)
            x_el = (xa, xb)
            y_el = (ya, yb)
            
            # Check if this point conflicts with any point in S
            conflict = False
            for s_idx in S:
                sa, sb, sc, sd = idx_to_point(s_idx)
                sx_el = (sa, sb)
                sy_el = (sc, sd)
                
                # Check direction (0,1): same x
                if x_el == sx_el:
                    conflict = True
                    break
                
                # Check direction (1, m) for some m
                # Need y - sy = m * (x - sx) for some m
                # If x == sx, already handled above
                dx = sub_el(x_el, sx_el)
                dy = sub_el(y_el, sy_el)
                # If dx == 0, handled. Otherwise m = dy * dx^{-1}
                # Check if such m exists and if the line through these two points in direction (1,m)
                # is a valid direction. It always is, since m is in F_49.
                # So any two distinct points with different x are on a line of direction (1, m)
                # for a unique m.
                if dx[0] != 0 or dx[1] != 0:
                    conflict = True
                    break
            
            if not conflict:
                S.add(idx)
        
        if len(S) > len(best_S):
            best_S = S.copy()
    
    # Build N = complement of S
    S_arr = np.zeros(total, dtype=bool)
    for idx in best_S:
        S_arr[idx] = True
    
    N_indices = np.where(~S_arr)[0]
    
    # Convert to array of shape (k, 2, 2)
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i] = [[xa, xb], [ya, yb]]
    
    return result
