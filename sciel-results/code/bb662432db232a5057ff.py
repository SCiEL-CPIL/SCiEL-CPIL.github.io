import numpy as np
import time
from itertools import combinations

def _score(points):
    """Calculate the maximum multiplicity of a positive doubled area."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    if n < 3:
        return 0
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            v1 = p2 - p1
            if j + 1 < n:
                rest = points[j+1:]
                terms = v1[0] * rest[:, 1] - v1[1] * rest[:, 0] + (v1[1]*p1[0] - v1[0]*p1[1])
                areas = np.abs(terms)
                for a in areas:
                    if a > 0:
                        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _score_two_line(B, T):
    """
    Score a two-line configuration where B and T are sorted lists of x-coordinates
    for bottom (y=0) and top (y=1) lines.
    
    For each difference k, the count is:
    - (count of pairs in B with diff k) * len(T)  [2 on bottom, 1 on top]
    - (count of pairs in T with diff k) * len(B)  [2 on top, 1 on bottom]
    - (count of pairs in B with diff k) * len(T)  [wait, no]
    
    Let me re-derive:
    Case 1: 2 points on bottom (y=0), 1 on top (y=1).
      P1=(b_i,0), P2=(b_j,0), P3=(t_k,1)
      Area = 0.5 * |b_i - b_j| * 1
      Doubled area = |b_i - b_j|
      For each pair (i,j) in B with difference k, and each t in T, we get one triangle.
      Count = (number of pairs in B with diff k) * |T|
    
    Case 2: 2 points on top (y=1), 1 on bottom (y=0).
      P1=(t_i,1), P2=(t_j,1), P3=(b_k,0)
      Area = 0.5 * |t_i - t_j| * 1
      Doubled area = |t_i - t_j|
      Count = (number of pairs in T with diff k) * |B|
    
    Case 3: 1 on bottom, 2 on top.
      P1=(b_i,0), P2=(t_j,1), P3=(t_k,1)
      Det = (t_j - b_i) * 1 - 1 * (t_k - b_i) = t_j - t_k
      Doubled area = |t_j - t_k|
      For each b in B, and each pair (j,k) in T with diff k, we get one triangle.
      Count = |B| * (number of pairs in T with diff k)
    
    Case 4: 2 on bottom, 1 on top - already covered in Case 1.
    Case 5: 1 on bottom, 1 on top, 1 on bottom - same as Case 1.
    
    Wait, I'm double counting. Let me be more careful.
    
    Any triple of 3 distinct points from the 30 points.
    The triple can have:
    - 3 on bottom (impossible, all y=0, area=0)
    - 3 on top (impossible, all y=1, area=0)
    - 2 on bottom, 1 on top
    - 1 on bottom, 2 on top
    
    Case 2B1T: 2 on bottom, 1 on top
      P1=(b_i,0), P2=(b_j,0), P3=(t_k,1)
      Doubled area = |b_i - b_j|
      For each pair (i,j) in B with diff k, and each t_k in T, we get one triangle.
      Count = (number of unordered pairs in B with diff k) * |T|
    
    Case 1B2T: 1 on bottom, 2 on top
      P1=(b_i,0), P2=(t_j,1), P3=(t_k,1)
      Doubled area = |t_j - t_k|
      For each b_i in B, and each pair (j,k) in T with diff k, we get one triangle.
      Count = |B| * (number of unordered pairs in T with diff k)
    
    So total count for doubled area k:
    count(k) = (pairs_B(k)) * |T| + |B| * (pairs_T(k))
    
    where pairs_B(k) is the number of unordered pairs in B with difference k,
    and pairs_T(k) is the number of unordered pairs in T with difference k.
    """
    B_set = set(B)
    T_set = set(T)
    nB = len(B)
    nT = len(T)
    
    # Count pairs in B with each difference
    pairs_B = {}
    for i in range(nB):
        for j in range(i + 1, nB):
            d = abs(B[i] - B[j])
            if d > 0:
                pairs_B[d] = pairs_B.get(d, 0) + 1
    
    # Count pairs in T with each difference
    pairs_T = {}
    for i in range(nT):
        for j in range(i + 1, nT):
            d = abs(T[i] - T[j])
            if d > 0:
                pairs_T[d] = pairs_T.get(d, 0) + 1
    
    # Total count for each doubled area k
    counts = {}
    for k in pairs_B:
        counts[k] = counts.get(k, 0) + pairs_B[k] * nT
    for k in pairs_T:
        counts[k] = counts.get(k, 0) + pairs_T[k] * nB
    
    if not counts:
        return 0
    return max(counts.values())

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy: Optimize two-line configurations.
    # Start with the 15x2 grid (contiguous x-coordinates).
    # Then use local search to perturb the x-coordinates to improve the score.
    
    # Initial configuration: 15 points on y=0 with x in [0,14], 15 on y=1 with x in [0,14]
    B0 = list(range(15))
    T0 = list(range(15))
    
    # Center to be in [-30, 30]
    B0 = [x - 7 for x in B0]
    T0 = [x - 7 for x in T0]
    
    best_B = list(B0)
    best_T = list(T0)
    best_score = _score_two_line(best_B, best_T)
    
    # Local search: try to improve by moving individual x-coordinates
    # We'll use a greedy approach: try all small moves and take the best improvement.
    
    # Helper to check if a set of x-coordinates is valid (distinct, in bounds)
    def valid_x_set(xs, other_set=None):
        if len(xs) != 15:
            return False
        if len(set(xs)) != 15:
            return False
        for x in xs:
            if not (-30 <= x <= 30):
                return False
        if other_set is not None:
            # No constraint between B and T for now (they can overlap in x)
            pass
        return True
    
    # Local search loop
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try moves on B
        for i in range(15):
            if time.time() > deadline - 5:
                break
            old_val = best_B[i]
            # Try small deltas
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < -30 or new_val > 30:
                    continue
                if new_val in best_B:
                    continue
                # Try the move
                best_B[i] = new_val
                if valid_x_set(best_B):
                    s = _score_two_line(best_B, best_T)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_B[i] = old_val
        
        # Try moves on T
        for i in range(15):
            if time.time() > deadline - 5:
                break
            old_val = best_T[i]
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < -30 or new_val > 30:
                    continue
                if new_val in best_T:
                    continue
                best_T[i] = new_val
                if valid_x_set(best_T):
                    s = _score_two_line(best_B, best_T)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_T[i] = old_val
    
    # Also try some random restarts if we have time
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Generate a random two-line configuration
            # Sample 15 distinct x-coordinates for B and 15 for T
            B_rand = list(np.random.choice(range(-30, 31), size=15, replace=False))
            T_rand = list(np.random.choice(range(-30, 31), size=15, replace=False))
            
            s = _score_two_line(B_rand, T_rand)
            if s > best_score:
                best_score = s
                best_B = B_rand
                best_T = T_rand
            
            # Local search from this starting point
            cur_B = list(B_rand)
            cur_T = list(T_rand)
            cur_score = s
            
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                # Try a random move
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, 15)
                    old = cur_B[idx]
                    new = np.random.randint(-30, 31)
                    if new == old or new in cur_B:
                        continue
                    cur_B[idx] = new
                    if valid_x_set(cur_B):
                        s2 = _score_two_line(cur_B, cur_T)
                        if s2 > cur_score:
                            cur_score = s2
                            if s2 > best_score:
                                best_score = s2
                                best_B = cur_B
                                best_T = cur_T
                        else:
                            cur_B[idx] = old
                else:
                    idx = np.random.randint(0, 15)
                    old = cur_T[idx]
                    new = np.random.randint(-30, 31)
                    if new == old or new in cur_T:
                        continue
                    cur_T[idx] = new
                    if valid_x_set(cur_T):
                        s2 = _score_two_line(cur_B, cur_T)
                        if s2 > cur_score:
                            cur_score = s2
                            if s2 > best_score:
                                best_score = s2
                                best_B = cur_B
                                best_T = cur_T
                        else:
                            cur_T[idx] = old
    
    # Convert best B, T to points
    best_points = [(x, 0) for x in best_B] + [(x, 1) for x in best_T]
    
    # Final verification with the original _score function
    final_score = _score(best_points)
    
    # Fallback: if something went wrong, use the standard grid
    if final_score < 420:
        best_points = [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]
        final_score = _score(best_points)
    
    return best_points
