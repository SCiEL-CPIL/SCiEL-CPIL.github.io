import random
import time
import itertools
from typing import List, Tuple
from collections import defaultdict

def construct():
    """
    Returns 20 distinct integer points [x, y] with |x|, |y| <= 100
    such that no four are collinear.
    Aims to minimize the largest subset with no three collinear
    by maximizing the number of 3-collinear triples.
    
    Approach: Greedy Construction followed by Local Search.
    1. Greedily build a set of 20 points. At each step, pick the point
       that maximizes the number of new collinear triples formed, while
       ensuring no 4 points are collinear.
    2. Refine using local search (hill climbing) to further maximize
       collinear triples (which minimizes the target score).
    """
    N = 20
    BOUND = 50  # Reduced bound for candidate pool density, still within 100
    
    # Helper: Normalize line representation (A, B, C) for Ax + By + C = 0
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        # Normalize
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        # Make gcd 1
        from math import gcd
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        return (A, B, C)

    def is_on_line(p, line_key):
        A, B, C = line_key
        return A * p[0] + B * p[1] + C == 0

    def count_three_collinear(points):
        count = 0
        # Use line keys to count
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        for count_pairs in line_counts.values():
            # Number of triples is C(count_pairs, 2)
            if count_pairs >= 2:
                count += count_pairs * (count_pairs - 1) // 2
        return count

    def is_valid_no_four_collinear(points):
        # Check if any line has >= 4 points
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        for count_pairs in line_counts.values():
            if count_pairs >= 3: # 3 pairs means 4 points? No.
                # If 4 points are collinear, they form C(4,2)=6 pairs on that line.
                # If 3 points are collinear, they form C(3,2)=3 pairs.
                # We want to detect 4 collinear.
                # If line has k points, pairs = k*(k-1)/2.
                # We need to check if k >= 4.
                # We can just check max points on any line.
                pass
        # Simpler check for validity:
        # If any line has 4 or more points, it's invalid.
        # We can compute max points on a line.
        max_pts = 0
        line_pts = defaultdict(list)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                if len(line_pts[key]) < 2:
                    line_pts[key].append(points[i])
                    line_pts[key].append(points[j])
                else:
                    # Check if points[i] or points[j] already in list?
                    # This is inefficient.
                    pass
        
        # Efficient validity check:
        # For each point p, check lines formed with other points.
        # If any line has 3 other points, it's invalid.
        for i in range(len(points)):
            p = points[i]
            line_counts = defaultdict(int)
            for j in range(len(points)):
                if i == j: continue
                key = get_line_key(p, points[j])
                line_counts[key] += 1
            for c in line_counts.values():
                if c >= 3:
                    return False
        return True

    # Greedy Construction
    def greedy_construct():
        points = []
        # Candidate pool: Grid
        candidates = [[x, y] for x in range(-BOUND, BOUND + 1) for y in range(-BOUND, BOUND + 1)]
        random.shuffle(candidates)
        
        # Line counts for current points
        line_counts = defaultdict(int)
        point_to_lines = defaultdict(list) # Not strictly needed if we recompute, but helps
        
        for _ in range(N):
            best_candidate = None
            best_gain = -1
            
            # To avoid O(M * N^2) being too slow if M is large, we sample or limit candidates
            # M = 100*100 = 10000. N=20.
            # For each candidate, check validity and gain.
            # Validity: For each existing point q, line(p,q) count < 3.
            # Gain: Sum of C(count(line(p,q)), 2) for all q?
            # Actually, gain is number of new triples.
            # A new triple is (a,b,p) where a,b in points.
            # This is equivalent to: For each line L through p, let k be # points in points on L.
            # New triples = C(k, 2).
            
            # We can optimize by precomputing line counts for existing points.
            # But line counts change as we add points.
            
            # Let's just iterate through a subset of candidates if full set is too slow.
            # 10000 candidates * 20 existing points * 20 steps = 4,000,000 ops. Fast enough.
            
            for p in candidates:
                # Check validity
                valid = True
                gain = 0
                line_counts_p = defaultdict(int)
                
                for q in points:
                    key = get_line_key(p, q)
                    line_counts_p[key] += 1
                
                # Check validity: no line has >= 3 existing points
                for c in line_counts_p.values():
                    if c >= 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate gain
                for c in line_counts_p.values():
                    if c >= 2:
                        gain += c * (c - 1) // 2
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = p
            
            if best_candidate is None:
                # No valid candidate found? Should not happen with large bound.
                # Fallback: pick random valid point
                for p in candidates:
                    valid = True
                    line_counts_p = defaultdict(int)
                    for q in points:
                        key = get_line_key(p, q)
                        line_counts_p[key] += 1
                    for c in line_counts_p.values():
                        if c >= 3:
                            valid = False
                            break
                    if valid:
                        best_candidate = p
                        break
            
            if best_candidate is None:
                raise ValueError("Failed to construct valid set")
            
            points.append(best_candidate)
            
            # Update line counts? 
            # We recompute in next iteration, so no need to maintain globally if we recompute gain.
            # But we need to ensure we don't pick duplicates.
            # Since we pick from candidates, we should remove picked point from candidates?
            # Or just check distinctness.
            # Greedy might pick same point again? 
            # The problem asks for distinct points.
            # We should ensure distinctness.
            
            # Remove picked point from candidates to avoid duplicates
            candidates.remove(best_candidate)

        return points

    # Local Search
    def local_search(points):
        best_points = [p[:] for p in points]
        best_score = count_three_collinear(best_points)
        
        start_time = time.time()
        time_limit = 50 # Leave buffer for construction
        
        while time.time() - start_time < time_limit:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                x, y = best_points[i]
                # Try moving point i to nearby positions
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        
                        # Check distinctness
                        new_point = [nx, ny]
                        if new_point in best_points:
                            continue
                        
                        new_points = best_points.copy()
                        new_points[i] = new_point
                        
                        # Check validity (no 4 collinear)
                        if not is_valid_no_four_collinear(new_points):
                            continue
                        
                        new_score = count_three_collinear(new_points)
                        if new_score > best_score:
                            best_points = new_points
                            best_score = new_score
                            improved = True
                            x, y = nx, ny # Update for next neighbor check
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_points

    # Main
    start_time = time.time()
    best_points = greedy_construct()
    
    # Verify validity of constructed points
    if not is_valid_no_four_collinear(best_points):
        # Fallback to random if greedy fails (unlikely)
        while True:
            pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(N)]
            if is_valid_no_four_collinear(pts) and len(set(map(tuple, pts))) == N:
                best_points = pts
                break

    # Refine
    best_points = local_search(best_points)
    
    # Final validity check
    if not is_valid_no_four_collinear(best_points) or len(set(map(tuple, best_points))) != N:
        # This should not happen, but return a safe fallback if it does
        # Generate a simple valid set (e.g. points on a curve with no 4 collinear)
        # y = x^2 mod something?
        # Just use a grid with shifts
        pts = []
        for i in range(20):
            pts.append([i, i*i % 100])
        # Ensure distinct and valid
        # This might have 4 collinear?
        # Parabola has no 3 collinear. So valid.
        # But score will be 20.
        # We prefer the local search result.
        pass

    return best_points
