# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    their maximum squared norm is strictly less than their minimum pairwise squared distance.
    Aims to maximize the number of points (k).
    """
    d = 11
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 10  # seconds
    
    # We want to find a set of points P such that:
    # 1. ||p_i||^2 <= M for all i
    # 2. ||p_i - p_j||^2 > M for all i != j
    # This is equivalent to finding a packing of spheres of radius sqrt(M)/2 centered at these points,
    # where the spheres are inside the ball of radius sqrt(M) around the origin? No.
    # Let R^2 = max ||p_i||^2.
    # Condition: min_{i!=j} ||p_i - p_j||^2 >= R^2 + 1 (since integers, strict > R^2 means >= R^2+1 if R^2 is integer?
    # The task says "minimum squared distance ... is >= the maximum squared norm".
    # So min_dist_sq >= max_norm_sq.
    
    # Strategy:
    # Fix a target max_norm_sq = N.
    # We want to pack as many integer points with norm^2 <= N as possible,
    # such that all pairwise distances squared are >= N.
    # Note: If two points have norm^2 <= N, their distance squared is ||p||^2 + ||q||^2 - 2 p.q.
    # To maximize distance, we want p.q to be small (negative).
    
    # This is a hard combinatorial problem.
    # Let's try a greedy approach with local search.
    # Start with a small set, try to add points, try to swap.
    
    # Candidate pool:
    # Points with small norms.
    # Let's generate points with norm^2 <= 4 or 5?
    # If N=1, only +/- e_i and 0. 0 is excluded.
    # Points with norm^2=1: 22 points (+-e_i).
    # Distance between e_i and e_j (i!=j) is 2. Distance between e_i and -e_i is 4.
    # Max norm^2 = 1. Min dist^2 = 2. 2 >= 1. Valid.
    # Can we add more?
    # If we add a point with norm^2=2, e.g., e_1+e_2.
    # Dist to e_1 is 1. 1 < 2 (new max norm^2). Invalid.
    # So if we include norm^2=2 points, max norm^2 becomes 2.
    # Then min dist^2 must be >= 2.
    # Dist(e_1+e_2, e_1) = 1 < 2. So e_1 cannot be in the set if e_1+e_2 is.
    
    # Let's try to construct a set with max norm^2 = 2.
    # Points must have norm^2 <= 2.
    # Candidates:
    # Norm^2=1: 22 points (e_i, -e_i).
    # Norm^2=2: points with two non-zero coords, each +/-1.
    # Number of such points: C(11,2) * 2^2 = 55 * 4 = 220.
    # Total candidates: 242.
    
    # We want to select a subset S of these 242 points such that:
    # min_{p,q in S, p!=q} ||p-q||^2 >= 2.
    # Since max norm^2 is 2, the condition is min dist^2 >= 2.
    
    # Let's analyze distances between candidates.
    # Type A: Norm^2=1 (e_i, -e_i)
    # Type B: Norm^2=2 (e_i+/-e_j, etc)
    
    # Dist^2 between two Type A:
    # e_i, e_j (i!=j): 2. OK.
    # e_i, -e_i: 4. OK.
    # e_i, -e_j (i!=j): 2. OK.
    # So any two distinct Type A points have dist^2 >= 2.
    # Thus, we can include ALL 22 Type A points.
    
    # Now, can we include Type B points?
    # Let p in Type B. e.g., p = e_1 + e_2.
    # Dist^2(p, e_1) = ||e_2||^2 = 1. < 2. Conflict.
    # Dist^2(p, e_2) = 1. Conflict.
    # Dist^2(p, -e_1) = ||2e_1+e_2||^2 = 5. OK.
    # Dist^2(p, -e_2) = 5. OK.
    # Dist^2(p, e_3) (3!=1,2) = ||e_1+e_2-e_3||^2 = 3. OK.
    # So p conflicts with e_1 and e_2 (and -e_1? no, -e_1 is far).
    # Wait, p = e_1 + e_2.
    # q = e_1. dist^2 = 1.
    # q = -e_1. dist^2 = ||e_2||^2 + ||2e_1||^2? No.
    # (e_1+e_2) - (-e_1) = 2e_1 + e_2. Norm^2 = 4+1=5.
    # So p conflicts with e_1 and e_2.
    # What about p' = -e_1 - e_2?
    # p' conflicts with -e_1 and -e_2.
    
    # If we include ALL Type A points, we cannot include ANY Type B point,
    # because every Type B point e_i+/-e_j conflicts with e_i and e_j (or -e_i, -e_j).
    # Specifically, e_i+e_j conflicts with e_i and e_j.
    # e_i-e_j conflicts with e_i and -e_j.
    # Since all Type A points are present, any Type B point will be within distance 1 of some Type A point.
    # Thus, with max_norm^2=2 and all Type A points, we cannot add Type B points.
    # Size = 22.
    
    # Can we do better by dropping some Type A points to add more Type B points?
    # Suppose we drop e_1 and e_2.
    # We can add e_1+e_2.
    # But e_1+e_2 might conflict with other Type B points we want to add.
    # Let's check distances between Type B points.
    # p = e_1+e_2, q = e_1+e_3.
    # dist^2 = ||e_2-e_3||^2 = 2. OK.
    # p = e_1+e_2, q = e_1-e_2.
    # dist^2 = ||2e_2||^2 = 4. OK.
    # p = e_1+e_2, q = -e_1-e_2.
    # dist^2 = ||2e_1+2e_2||^2 = 8. OK.
    # p = e_1+e_2, q = e_2+e_3.
    # dist^2 = ||e_1-e_3||^2 = 2. OK.
    # p = e_1+e_2, q = -e_1+e_2.
    # dist^2 = ||2e_1||^2 = 4. OK.
    # p = e_1+e_2, q = e_1+e_2. Same point.
    # p = e_1+e_2, q = -e_1-e_2. Dist 8.
    # p = e_1+e_2, q = e_1-e_2. Dist 4.
    # p = e_1+e_2, q = -e_1+e_2. Dist 4.
    # p = e_1+e_2, q = -e_1-e_2.
    #
    # It seems Type B points are pairwise distance >= 2?
    # Let's verify.
    # p, q in Type B.
    # p = s1 e_i + s2 e_j. q = t1 e_k + t2 e_l.
    # If {i,j} and {k,l} are disjoint:
    #   dist^2 = ||p||^2 + ||q||^2 = 2+2=4 (if orthogonal).
    #   If not orthogonal, e.g. p=e1+e2, q=e3+e4. dist^2=4.
    # If they share 1 index, e.g. p=e1+e2, q=e1+e3.
    #   dist^2 = ||e2-e3||^2 = 2.
    #   p=e1+e2, q=e1-e3. dist^2 = ||e2+e3||^2 = 2.
    #   p=e1+e2, q=-e1+e3. dist^2 = ||2e1+e2-e3||^2 = 4+1+1=6.
    # If they share 2 indices (same support):
    #   p=e1+e2, q=e1+e2. Same.
    #   p=e1+e2, q=e1-e2. dist^2 = ||2e2||^2 = 4.
    #   p=e1+e2, q=-e1+e2. dist^2 = ||2e1||^2 = 4.
    #   p=e1+e2, q=-e1-e2. dist^2 = ||2e1+2e2||^2 = 8.
    #
    # So min dist^2 between distinct Type B points is 2.
    # Thus, the set of ALL Type B points is valid internally (min dist^2 >= 2).
    # Size of Type B set = 220.
    # Max norm^2 = 2.
    # Condition: min dist^2 >= 2.
    # So the set of all 220 Type B points is a valid solution!
    # k = 220.
    
    # Can we mix Type A and Type B?
    # We saw that Type B points conflict with Type A points at distance 1.
    # If we include a Type B point p, we must exclude all Type A points at distance 1 from p.
    # p = e1+e2 conflicts with e1 and e2.
    # So we lose 2 Type A points to gain 1 Type B point?
    # Ratio is bad. 220 Type B points vs 22 Type A points.
    # If we take all 220 Type B, we can take NO Type A?
    # Let's check if any Type A point is at distance >= 2 from ALL Type B points.
    # Take e1.
    # Dist(e1, e1+e2) = 1. Conflict.
    # So e1 conflicts with e1+e2, e1-e2, e1+e3, ...
    # Specifically, e1 conflicts with any Type B point involving e1.
    # There are 10 * 4 = 40 such points?
    # Supports containing 1: {1,2}..{1,11}. 10 supports.
    # Signs: 4 each. Total 40.
    # So e1 conflicts with 40 Type B points.
    # If we want to keep e1, we must exclude these 40 Type B points.
    # Gain 1, lose 40. Bad.
    # So it's better to take all Type B and no Type A.
    # k = 220.
    
    # Can we go to max_norm^2 = 3?
    # Candidates:
    # Type A: Norm^2=1 (22 pts)
    # Type B: Norm^2=2 (220 pts)
    # Type C: Norm^2=3 (e.g., e1+e2+e3, or e1+2e2? No, integer coords.
    # Norm^2=3:
    # 1^2+1^2+1^2 = 3. Supports of size 3. C(11,3)=165. Signs 2^3=8. Total 165*8=1320.
    # 1^2+sqrt(2)^2? No.
    # So Type C has 1320 points.
    
    # If we set max_norm^2 = 3, we need min dist^2 >= 3.
    # Let's check distances within Type C.
    # p=e1+e2+e3, q=e1+e2+e4.
    # dist^2 = ||e3-e4||^2 = 2. < 3. Conflict!
    # So we cannot take all Type C points.
    # In fact, we can't even take a large subset easily.
    
    # What about mixing Type B and Type C?
    # p in Type B (norm^2=2), q in Type C (norm^2=3).
    # dist^2 = 2+3 - 2 p.q = 5 - 2 p.q.
    # We need dist^2 >= 3 => 5 - 2 p.q >= 3 => 2 p.q <= 2 => p.q <= 1.
    # p = e1+e2. q = e1+e2+e3.
    # p.q = 1+1+0 = 2.
    # dist^2 = 5 - 4 = 1. < 3. Conflict.
    # So p and q conflict if they share 2 coordinates with same sign.
    # p=e1+e2, q=e1-e2+e3.
    # p.q = 1-1+0 = 0.
    # dist^2 = 5. OK.
    # p=e1+e2, q=-e1+e2+e3.
    # p.q = -1+1+0 = 0.
    # dist^2 = 5. OK.
    # p=e1+e2, q=e1+e3+e4.
    # p.q = 1.
    # dist^2 = 5-2=3. OK.
    
    # So Type B and Type C can coexist if they don't "overlap" too much.
    # However, the conflicts within Type C are severe (dist 2).
    # And Type B points are numerous (220).
    
    # Let's stick with the Type B solution (k=220) as a strong baseline.
    # Can we improve on 220?
    # Maybe a different norm?
    # What if we use points with larger norms but sparser?
    # The "kissing number" problem is related to sphere packing.
    # In 11 dimensions, the kissing number is 558 (for real numbers).
    # For integer points, it's different.
    # Our constraint is min_dist^2 >= max_norm^2.
    # This is like packing spheres of radius r=sqrt(max_norm^2)/2?
    # No, the spheres are centered at p_i and must not overlap?
    # If we place spheres of radius R/2 at p_i, where R^2 = max_norm^2,
    # then two spheres overlap if dist < R.
    # Our condition is dist >= R.
    # So the spheres are tangent or disjoint.
    # Also, all spheres must be inside the ball of radius R centered at origin?
    # No, the centers are inside the ball of radius R.
    # So we are packing spheres of radius R/2 inside the ball of radius 3R/2?
    # No.
    # The centers are in B(0, R).
    # The spheres are B(p_i, R/2).
    # The farthest point in a sphere is at distance ||p_i|| + R/2 <= R + R/2 = 1.5 R.
    # So we are packing spheres of radius R/2 in a container of radius 1.5 R.
    # The volume ratio is (1.5)^11 = 86.5.
    # The volume of one sphere is (R/2)^11.
    # So we can fit roughly 86.5 / (packing_efficiency) spheres.
    # Packing efficiency in 11D is low, maybe 1/100?
    # So maybe a few hundred points.
    # 220 seems reasonable.
    
    # Let's verify the Type B set construction.
    # d=11.
    # Points: all vectors with exactly two non-zero coordinates, each +/-1.
    # Count: C(11,2) * 4 = 55 * 4 = 220.
    # Max norm^2 = 1^2 + 1^2 = 2.
    # Min dist^2 between distinct points:
    # We established min dist^2 = 2.
    # Condition: 2 >= 2. Valid.
    
    # Can we add any point from Type A?
    # No, as discussed.
    # Can we add any point from Type C?
    # Max norm^2 would become 3.
    # Then min dist^2 must be >= 3.
    # But min dist^2 in Type B is 2.
    # So we would have to remove all Type B pairs that are at distance 2.
    # That's most of them.
    # So likely not worth it.
    
    # What if we use a different set entirely?
    # E.g., points with norm^2 = 11? (All coords +/-1).
    # Count: 2^11 = 2048.
    # Max norm^2 = 11.
    # Dist^2 between p, q (all +/-1):
    # ||p-q||^2 = sum (p_i-q_i)^2.
    # p_i-q_i is 0 if p_i=q_i, +/-2 if p_i!=q_i.
    # Let k be the number of coordinates where they differ.
    # dist^2 = 4k.
    # Min dist^2 is 4 (if they differ in 1 coordinate).
    # Condition: 4 >= 11? False.
    # So this set is invalid.
    
    # What if we take a subset of these?
    # We need dist^2 >= 11 => 4k >= 11 => k >= 3.
    # So we need a code of length 11, binary (or +/-1), with minimum Hamming distance 3.
    # This is a binary code with d_min=3.
    # Max size A(11, 3).
    # Hamming bound: 2^11 / (1 + 11 + 55) = 2048 / 67 ~ 30.
    # So max size is around 30.
    # Much worse than 220.
    
    # What about norm^2 = 4?
    # Points with 4 non-zero coords? Or 2 coords with 2?
    # Let's try points with exactly 2 non-zero coords, values +/-2?
    # Norm^2 = 4+4=8? No.
    # Values +/-2: norm^2 = 8.
    # Or one coord +/-2, others 0? Norm^2=4.
    # Points: 22 points (2 e_i, -2 e_i).
    # Dist^2 between 2e_i and 2e_j = 8.
    # Dist^2 between 2e_i and -2e_i = 16.
    # Min dist^2 = 8.
    # Max norm^2 = 4.
    # 8 >= 4. Valid.
    # k=22. Worse than 220.
    
    # What about mixing?
    # It seems the "2 non-zero coords, value 1" set is very good.
    
    # Is there a set with k > 220?
    # Let's try to search locally around the Type B set.
    # Can we swap some Type B points for others?
    # The Type B set is "saturated" in the sense that adding any other integer point
    # with norm^2 <= 2 is impossible (only Type A and B exist).
    # Adding Type A breaks the condition.
    # So we are stuck at 220 for max_norm^2=2.
    
    # What if we allow max_norm^2=3?
    # We need min dist^2 >= 3.
    # Let's try to construct a set.
    # Start with Type B (220 pts).
    # Remove points that are at distance 2 from each other?
    # No, we need to remove points to make room for Type C?
    # Or just find a better set.
    
    # Let's use a greedy algorithm to find a large set with max_norm^2 <= N.
    # We'll try N=2, 3, 4.
    
    def generate_candidates(N, d):
        """Generate all integer points with norm^2 <= N."""
        points = []
        # Simple enumeration for small N
        # Range for each coord: -sqrt(N) to sqrt(N)
        limit = int(np.sqrt(N)) + 1
        for coords in product(range(-limit, limit+1), repeat=d):
            norm_sq = sum(c*c for c in coords)
            if norm_sq <= N and norm_sq > 0:
                points.append(np.array(coords, dtype=np.int64))
        return np.array(points)

    def min_pairwise_dist_sq(points):
        """Calculate minimum pairwise squared distance."""
        k = len(points)
        if k < 2:
            return float('inf')
        min_d2 = float('inf')
        for i in range(k):
            for j in range(i+1, k):
                d2 = np.sum((points[i] - points[j])**2)
                if d2 < min_d2:
                    min_d2 = d2
        return min_d2

    def max_norm_sq(points):
        """Calculate maximum squared norm."""
        if len(points) == 0:
            return 0
        return int(np.max(np.sum(points**2, axis=1)))

    # Greedy construction
    best_k = 0
    best_points = np.array([np.ones(d), -1*np.ones(d)], dtype=np.int64)
    
    for N in [2, 3, 4, 5]:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        candidates = generate_candidates(N, d)
        # Sort candidates by norm, then lexicographically
        norms = np.sum(candidates**2, axis=1)
        idx = np.argsort(norms)
        candidates = candidates[idx]
        
        selected = []
        # Greedy: add point if it doesn't violate min_dist >= N
        # Note: The condition is min_dist >= max_norm.
        # If we fix max_norm = N, we need min_dist >= N.
        # But if we add a point with norm < N, the max_norm might be N (if we have any point with norm N).
        # So we require all pairwise distances >= N.
        
        for p in candidates:
            if time.time() - start_time > time_limit - safety_margin:
                break
            valid = True
            for q in selected:
                d2 = np.sum((p - q)**2)
                if d2 < N:
                    valid = False
                    break
            if valid:
                selected.append(p)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = np.array(selected, dtype=np.int64)
            
        # Local search / Swap
        # Try to swap points in selected to allow more points?
        # This is complex. For now, greedy might be enough.
        # The Type B set for N=2 should be found by greedy?
        # Candidates for N=2:
        # Norm 1: 22 pts.
        # Norm 2: 220 pts.
        # Greedy adds Norm 1 first?
        # If it adds e1, then e2... it adds all 22 Norm 1 points.
        # Then it tries Norm 2 points.
        # e1+e2 is dist 1 from e1. Rejected.
        # So it ends up with 22 points.
        # But the optimal for N=2 is 220 (all Norm 2 points).
        # So greedy with fixed order fails.
        
        # We need a better heuristic.
        # Try greedy with different orders?
        # Or specifically construct the Type B set.
        
        # Let's explicitly construct the Type B set (Norm 2, 2 non-zero coords).
        if N == 2:
            type_b = []
            for i in range(d):
                for j in range(i+1, d):
                    for si in [-1, 1]:
                        for sj in [-1, 1]:
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = si
                            p[j] = sj
                            type_b.append(p)
            type_b = np.array(type_b, dtype=np.int64)
            if len(type_b) > best_k:
                # Verify validity
                # Max norm is 2.
                # Min dist is 2.
                # 2 >= 2.
                best_k = len(type_b)
                best_points = type_b

    # Check if we can do better with N=3?
    # Let's try a random restart greedy for N=3.
    # Or just return the best found so far.
    
    # The Type B set (220) is a solid result.
    # Let's double check if we can improve N=3.
    # If we use a code-like structure.
    # But 220 is already quite high.
    
    # One more idea:
    # What if we use points with norm^2=1?
    # Max size 22.
    # Norm^2=2?
    # Max size 220.
    # Norm^2=3?
    # Let's see if we can find a set of size > 220 with max_norm^2=3.
    # This requires min_dist^2 >= 3.
    # The Type B set has min_dist^2=2.
    # So we must remove points.
    # If we remove enough Type B points to raise min_dist^2 to 3, can we add Type C points?
    # Unlikely to beat 220.
    
    # Let's return the best points.
    return best_points

# EVOLVE-BLOCK-END
