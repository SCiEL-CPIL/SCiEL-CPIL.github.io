# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position using a dynamic program
    over convex chains.
    
    Points are sorted lexicographically. For each pivot i, later points (j > i) are
    sorted by atan2 angle around i. A turn p->q->r counts as convex if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    
    We count 7-gons by: for each 7-tuple of points in convex position, the "smallest"
    point in lexicographic order is the pivot, and the remaining 6 form a convex chain.
    Actually, we need to be more careful. Let's use the standard approach:
    
    For each pair (i, j) with i < j, we count the number of convex chains of length L
    starting with i, j (where all subsequent points have index > j). A 7-gon is counted
    when we have a convex chain of 6 points (i, j, p3, p4, p5, p6) such that the turn
    p6 -> i is also convex (closing the polygon).
    
    Actually, the standard DP: dp[i][j][k] = number of convex chains of length k
    ending with points i, j (i < j, and the chain goes i, ..., j in angular order around
    some earlier point... this is getting complex).
    
    Let me use a cleaner formulation:
    Sort points lexicographically: P[0], P[1], ..., P[n-1].
    For each pivot P[i], consider points P[j] with j > i. Sort them by angle around P[i].
    Let the sorted order be Q[0], Q[1], ..., Q[m-1] where Q[t] = P[perm[t]].
    
    A convex chain of length L starting at P[i] and going through Q[a], Q[b], ... 
    means P[i], Q[a], Q[b], ... forms a convex chain where each consecutive triple
    makes a left turn (convex turn).
    
    dp[i][a][k] = number of convex chains of length k starting at P[i], second point Q[a],
    with all subsequent points having larger index in the Q ordering.
    
    A 7-gon with pivot P[i] consists of a convex chain of length 7: P[i], Q[a1], Q[a2], ..., Q[a6]
    where the chain is convex (each turn is left) AND the closing turn Q[a6] -> P[i] -> Q[a1] 
    is also convex. Wait, the closing is Q[a6] -> P[i] and then P[i] -> Q[a1], but we need
    the turn at P[i] to be convex too, i.e., the turn Q[a6] -> P[i] -> Q[a1] should be a left turn.
    
    Actually for a convex polygon, all turns are left (or all right). If we go around the
    polygon in order, each consecutive triple makes a left turn. So for the 7-gon 
    P[i], Q[a1], Q[a2], ..., Q[a6], we need:
    - Turn P[i] -> Q[a1] -> Q[a2] is left
    - Turn Q[a1] -> Q[a2] -> Q[a3] is left
    - ...
    - Turn Q[a5] -> Q[a6] -> P[i] is left  (closing turn)
    
    The DP counts chains where all internal turns are left. We then check the closing turn.
    
    Let's define:
    dp[a][k] = number of convex chains of length k starting at Q[a] (where Q is the angular
    sorted list for pivot i), such that all points in the chain have index > a in Q order,
    and all turns are left.
    
    Base: dp[a][1] = 1 for all a (chain of length 1 is just Q[a])
    dp[a][2] = number of b > a such that turn P[i] -> Q[a] -> Q[b] is left... 
    
    Hmm, this is tricky because the first turn involves the pivot. Let me reconsider.
    
    Alternative: dp[a][k] = number of convex chains of length k where the first point is Q[a],
    the second point is some Q[b] with b > a, and all turns within the chain are left.
    But we also need the turn from P[i] to Q[a] to Q[b] to be left.
    
    Let me use: dp[a][b][k] = number of convex chains of length k starting with P[i], Q[a], Q[b]
    (a < b in Q order), where all turns are left.
    
    This is O(n^4) per pivot which is too slow.
    
    Let me use a simpler but correct O(n^4) approach:
    For each 7-tuple, check if it's convex. That's O(n^7) which is way too slow for n=33.
    
    n=33, C(33,7) = 4,272,048. That's actually manageable! Let's just brute force.
    4.27 million 7-tuples, for each we check 7 turns. That's about 30 million operations,
    which should be fast in NumPy if vectorized, or even in pure Python with optimization.
    """
    n = len(points)
    count = 0
    for i in range(n - 6):
        for j in range(i + 1, n - 5):
            for k in range(j + 1, n - 4):
                for l in range(k + 1, n - 3):
                    for m in range(l + 1, n - 2):
                        for o in range(m + 1, n - 1):
                            for p in range(o + 1, n):
                                # Check if points[i,j,k,l,m,o,p] form a convex 7-gon
                                pts = [points[i], points[j], points[k], points[l], points[m], points[o], points[p]]
                                convex = True
                                for t in range(7):
                                    a = pts[t]
                                    b = pts[(t+1) % 7]
                                    c = pts[(t+2) % 7]
                                    cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
                                    if cross <= 1e-9:
                                        convex = False
                                        break
                                if convex:
                                    count += 1
    return count

def _is_valid(points: np.ndarray) -> bool:
    """Check validity constraints."""
    n = len(points)
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check minimum distance
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            if dx*dx + dy*dy < 1e-12:
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                val = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                if abs(val) < 1e-9:
                    return False
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres problem: place 33 points in the unit square to minimize
    the number of convex 7-gons.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill
    
    np.random.seed(42)
    
    # Start with a random configuration
    n = 33
    best_points = np.random.rand(n, 2)
    if not _is_valid(best_points):
        # Adjust to make valid
        for _ in range(1000):
            best_points = np.random.rand(n, 2)
            if _is_valid(best_points):
                break
    
    best_count = _count_convex7(best_points)
    best_value = 1.0 / (1.0 + best_count)
    
    # Simulated annealing
    current_points = best_points.copy()
    current_count = best_count
    current_value = best_value
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(n)
        # Generate a small perturbation
        scale = 0.02 * T / 0.05  # Scale perturbation with temperature
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        old_x = current_points[idx, 0]
        old_y = current_points[idx, 1]
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Clamp to unit square
        new_x = np.clip(new_x, 0.001, 0.999)
        new_y = np.clip(new_y, 0.001, 0.999)
        
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        # Check validity
        if not _is_valid(current_points):
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
            T *= cooling
            continue
        
        new_count = _count_convex7(current_points)
        new_value = 1.0 / (1.0 + new_count)
        
        delta = new_value - current_value
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            current_value = new_value
            if new_value > best_value:
                best_value = new_value
                best_points = current_points.copy()
        else:
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_points

# EVOLVE-BLOCK-END
