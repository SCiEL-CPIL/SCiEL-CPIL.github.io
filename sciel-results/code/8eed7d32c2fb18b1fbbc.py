import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin for safety

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        for m in range(n):
                            if m != i and m != j and m != k:
                                if are_collinear(points[i], points[j], points[m]):
                                    return True
        return False

    def max_no_3_collinear_exact(points):
        """
        Exact branch and bound for largest subset with no 3 collinear.
        Optimized with bitmasks and precomputed conflict masks.
        """
        n = len(points)
        
        # Precompute lines and conflict masks
        # conflict_mask[i] is a bitmask of points j such that there exists k 
        # where i, j, k are collinear. 
        # Actually, for "no 3 collinear", adding i to S is invalid if there exist 
        # j, k in S such that i, j, k are collinear.
        # This is equivalent to: for any j in S, if i and j define a line L, 
        # are there any other points in S on L?
        
        # Let's precompute for each pair (i, j) the set of k collinear with i, j.
        # Then, when adding i to S, we check if for any j in S, 
        # (collinear_with(i, j) & S_mask) has any bit set other than i and j?
        # No, if k is in S and collinear with i, j, then {i, j, k} is a bad triple.
        
        # Let's use a simpler check for n=20:
        # O(n^2) check per addition. 20^2 = 400 ops. 
        # B&B depth 20. Branching factor ~2.
        # Worst case 2^20 * 400 = 400M ops. Too slow in Python if not pruned well.
        
        # Better: Precompute for each i, a list of pairs (j, k) that are collinear with i?
        # No, precompute for each i, a bitmask 'bad_with[i]' such that 
        # 'bad_with[i] & S_mask' is not empty iff adding i creates a bad triple?
        # This is not quite right because the bad triple involves 2 points from S.
        
        # Let's stick to the O(n^2) check but optimize the loop.
        # And use a strong lower bound for pruning.
        
        best = 0
        points_tuple = [tuple(p) for p in points]
        
        # Heuristic ordering: points involved in more collinear triples first?
        # Let's just use index order for simplicity, or sort by "density".
        
        def backtrack(S, start_idx, S_mask):
            nonlocal best
            current_len = len(S)
            
            # Pruning: max possible size
            if current_len + (n - start_idx) <= best:
                return
                
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip start_idx
            backtrack(S, start_idx + 1, S_mask)
            
            # Option 2: Include start_idx
            p_idx = start_idx
            valid = True
            # Check if p_idx is collinear with any pair in S
            for j in S:
                # Check if there is k in S such that p_idx, j, k are collinear
                # We can iterate over S again
                for k in S:
                    if k == j: continue
                    if are_collinear(points_tuple[p_idx], points_tuple[j], points_tuple[k]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(p_idx)
                backtrack(S, start_idx + 1, S_mask | (1 << p_idx))
                S.pop()
                
        backtrack([], 0, 0)
        return best

    def approximate_score(points):
        """
        Fast heuristic for the score.
        We want to MINIMIZE the largest subset with no 3 collinear.
        This is equivalent to MAXIMIZING the "clustering" of collinear triples.
        
        A simple proxy: 
        Count the number of collinear triples.
        More triples -> likely smaller independence number?
        Actually, we can compute a greedy upper bound on the independence number.
        Greedy Independent Set on the 3-uniform hypergraph of collinear triples:
        1. Start with all points.
        2. While there is a collinear triple (i, j, k) in the set:
           Remove the point with the highest "degree" (number of triples it's part of) 
           or just remove one of them arbitrarily?
           Actually, to find the LARGEST independent set, we should remove the "most constrained" point?
           No, greedy independent set usually adds points.
           
        Let's try a different proxy:
        The score is the size of the largest subset S such that no 3 points in S are collinear.
        This is the independence number of the 3-uniform hypergraph H where hyperedges are collinear triples.
        
        A fast approximation:
        1. Find all collinear triples.
        2. If there are no triples, score = 20.
        3. Use a greedy algorithm to estimate the independence number:
           - Start with empty set S.
           - Consider points in order of "least involvement" in triples?
           - Or, consider points in order of "most involvement" and REMOVE them?
           
        Let's use a simple random sampling approach for the proxy:
        Sample 100 random subsets of size 10-15 and check if they are valid? No, that finds valid subsets, not the max.
        
        Better proxy:
        Compute the maximum number of points that can be placed on any line? No.
        
        Let's use a simple local search heuristic for the proxy score:
        Start with all 20 points.
        While there is a collinear triple (i, j, k):
            Remove the point among i, j, k that is part of the most other triples.
        The remaining set size is a lower bound on the independence number?
        No, removing points from a valid set keeps it valid. So the remaining set is a valid "no 3 collinear" set.
        Thus, the size of the remaining set is a LOWER BOUND on the score (independence number).
        Wait, the task is to MINIMIZE the score (largest subset).
        So we want a configuration where this lower bound is as small as possible?
        No, the score IS the largest subset. If I find a valid subset of size 10, the score is at least 10.
        If I find a configuration where the greedy removal leaves only 8 points, it means there exists a subset of size 8 with no 3 collinear.
        But the actual score might be higher (e.g., 12).
        
        So this proxy gives a LOWER BOUND on the score.
        We want to MINIMIZE the score.
        So we want to find a configuration where even the greedy lower bound is small?
        No, that's not right. If the greedy lower bound is 8, the score is >= 8.
        If another config has greedy lower bound 10, its score is >= 10.
        So minimizing the greedy lower bound is a good heuristic for minimizing the actual score.
        
        Algorithm for proxy:
        1. Find all collinear triples.
        2. Count degree of each point (number of triples it's in).
        3. Remove the point with max degree.
        4. Update triples (remove any triple containing the removed point).
        5. Repeat until no triples left.
        6. Return the number of points remaining.
        
        This is O(T * N) where T is number of triples. For 20 points, max triples is C(20,3) = 1140.
        Very fast.
        """
        n = len(points)
        # Find all triples
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Degree of each point
        degree = [0] * n
        for t in triples:
            degree[t[0]] += 1
            degree[t[1]] += 1
            degree[t[2]] += 1
            
        active = [True] * n
        remaining_triples = set(triples)
        
        while remaining_triples:
            # Find point with max degree among active points
            max_deg = -1
            max_idx = -1
            for i in range(n):
                if active[i] and degree[i] > max_deg:
                    max_deg = degree[i]
                    max_idx = i
                    
            if max_idx == -1:
                break
                
            # Remove max_idx
            active[max_idx] = False
            new_triples = set()
            for t in remaining_triples:
                if max_idx in t:
                    # Update degrees for other points in t
                    for idx in t:
                        if idx != max_idx:
                            degree[idx] -= 1
                else:
                    new_triples.add(t)
            remaining_triples = new_triples
            
        # Count remaining active points
        return sum(active)

    # Initial Configuration: A dense grid to create many collinear triples.
    # 5x4 grid of points.
    # x in {0, 1, 2, 3, 4}, y in {0, 1, 2, 3}
    # Total 20 points.
    # Scale by 10 to stay within bounds? No, 0-4 is fine.
    # But we need |x|, |y| <= 100.
    # Let's center it.
    # x in {-2, -1, 0, 1, 2}, y in {-1.5, -0.5, 0.5, 1.5} -> not integer.
    # x in {-4, -2, 0, 2, 4}, y in {-3, -1, 1, 3}
    # This is a 5x4 grid with spacing 2.
    
    base_points = []
    for x in [-4, -2, 0, 2, 4]:
        for y in [-3, -1, 1, 3]:
            base_points.append((x, y))
            
    current_points = base_points[:]
    current_proxy = approximate_score(current_points)
    
    # Simulated Annealing
    # Move: Pick a random point, move it to a random nearby integer coordinate.
    # Or swap two points?
    # Moving one point is better for local exploration.
    
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    max_iters = 50000
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # Generate neighbor
        new_points = current_points[:]
        idx = random.randint(0, 19)
        ox, oy = new_points[idx]
        
        # Random step in [-3, 3]
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        nx = ox + dx
        ny = oy + dy
        
        if abs(nx) > 100 or abs(ny) > 100:
            continue
            
        # Check if new point already exists
        if (nx, ny) in new_points:
            continue
            
        new_points[idx] = (nx, ny)
        
        # Check 4 collinear constraint
        if has_4_collinear(new_points):
            continue
            
        new_proxy = approximate_score(new_points)
        
        # Delta is new_proxy - current_proxy
        # We want to MINIMIZE proxy, so delta < 0 is good.
        delta = new_proxy - current_proxy
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_points = new_points
            current_proxy = new_proxy
            
        T *= alpha
        if T < T_min:
            T = T_min

    # Final verification with exact B&B on the best found configuration
    # It's possible the proxy is misleading, so we should check a few candidates?
    # For now, just check the best one.
    # To be safer, we could keep the top 5 candidates by proxy and check them all.
    
    # Let's just check the best one.
    final_score = max_no_3_collinear_exact(current_points)
    
    # Fallback if something went wrong (e.g. timeout during B&B? No, B&B is at the end)
    # If B&B is too slow, we might time out.
    # 20 points B&B is usually fast if pruned.
    
    return [list(p) for p in current_points]

# EVOLVE-BLOCK-END
