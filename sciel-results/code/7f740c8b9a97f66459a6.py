import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
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

    def is_valid(points):
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def count_triples(points):
        """Count the number of collinear triples."""
        n = len(points)
        triples = 0
        line_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].append(i)
        
        # Reconstruct point sets for each line
        # Actually, the above loop adds each point to the line multiple times?
        # No, get_line(points[i], points[j]) is the line.
        # If we have points p1, p2, p3 collinear.
        # i=0, j=1 -> line L. line_points[L].append(0)
        # i=0, j=2 -> line L. line_points[L].append(0)
        # i=1, j=2 -> line L. line_points[L].append(1)
        # This is inefficient and messy.
        
        # Better:
        # For each pair (i, j), determine line.
        # We need to count distinct triples.
        # A triple (i, j, k) is collinear if line(i,j) == line(i,k).
        
        # O(n^3) is 8000 checks. Fast enough.
        for i in range(n):
            for j in range(i + 1, n):
                li = get_line(points[i], points[j])
                for k in range(j + 1, n):
                    if get_line(points[i], points[k]) == li:
                        triples += 1
        return triples

    def greedy_vertex_cover(points):
        """
        Approximate the minimum vertex cover size of the collinear triple hypergraph.
        Returns the size of the cover.
        Score = 20 - cover_size.
        """
        n = len(points)
        # Build hyperedges (triples)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                li = get_line(points[i], points[j])
                for k in range(j + 1, n):
                    if get_line(points[i], points[k]) == li:
                        triples.append((i, j, k))
        
        if not triples:
            return 0
            
        remaining_vertices = set(range(n))
        remaining_triples = [set(t) for t in triples]
        
        removed = 0
        while remaining_triples:
            counts = defaultdict(int)
            for t in remaining_triples:
                for v in t:
                    if v in remaining_vertices:
                        counts[v] += 1
            
            if not counts:
                break
                
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            v_to_remove = min(candidates)
            
            remaining_vertices.remove(v_to_remove)
            removed += 1
            
            new_remaining = []
            for t in remaining_triples:
                if v_to_remove not in t:
                    new_remaining.append(t)
            remaining_triples = new_remaining
            
        return removed

    def calc_score(points):
        cover = greedy_vertex_cover(points)
        return 20 - cover

    # Constructive Search
    best_score = 20
    best_points = None
    
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        points = []
        used_points = set()
        
        # Randomize the order of candidate generation or selection to explore different regions
        # We will try to add points one by one.
        
        for step in range(20):
            # Generate candidates
            # We want to pick a point that creates the most new triples.
            # But we also need to ensure validity (no 4 collinear).
            
            # To speed up, we don't check ALL 20000 points.
            # We check a random sample + points on lines defined by existing points.
            
            candidates = []
            
            # 1. Points on lines defined by existing pairs
            if len(points) >= 2:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        p1 = points[i]
                        p2 = points[j]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        # Extend the line
                        # Try small integer steps
                        for t in range(-5, 6):
                            if t == 0 or t == 1: continue
                            nx = p1[0] + t * dx
                            ny = p1[1] + t * dy
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                p = (nx, ny)
                                if p not in used_points:
                                    candidates.append(p)
            
            # 2. Random points
            for _ in range(100):
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in used_points:
                    candidates.append(p)
            
            # Deduplicate candidates
            candidates = list(set(candidates))
            random.shuffle(candidates)
            
            best_candidate = None
            best_new_triples = -1
            
            # Evaluate candidates
            # To make it faster, limit the number of candidates we fully evaluate
            # But for n=20, evaluating 200 candidates * 8000 ops = 1.6M ops. Might be slow if done 20*50 times.
            # 1000 * 1.6M = 1.6B ops. Too slow.
            
            # Optimization:
            # We can quickly filter candidates that create a 4-collinear violation.
            # And we can estimate triple increase faster.
            
            # Let's just try a smaller set of candidates per step.
            # Or, we can use a heuristic: 
            # Calculate "potential" triples for each candidate quickly?
            
            # Let's limit candidates to 50 per step to ensure speed.
            candidates = candidates[:50]
            
            for p in candidates:
                # Check validity
                test_points = points + [p]
                
                # Quick check for 4-collinear
                # For each pair (i, j) in test_points, check if line contains 4 points?
                # Or just check if p lies on a line with 3 existing points.
                # If p lies on a line with 3 existing points, adding p makes 4. Invalid.
                
                valid = True
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        li = get_line(points[i], points[j])
                        # Check if p is on li
                        A, B, C = li
                        if A * p[0] + B * p[1] + C == 0:
                            # Check if there is a 3rd point k on this line
                            # We need to know the count of points on this line among 'points'
                            # This requires maintaining line counts.
                            # For now, let's just do the full O(n^3) check for the candidate.
                            # But we can short-circuit.
                            pass
                
                # Full validity check is O(n^3).
                # Let's just run is_valid on test_points?
                # is_valid checks distinctness (p not in points is already known) and bounds.
                # And no 4 collinear.
                
                # To speed up, we can check if p creates a 4-collinear with existing points.
                # This is O(n^2).
                # Count points on each line defined by (p, existing_point).
                # If any line has 3 other points, it's invalid.
                
                line_counts_with_p = defaultdict(int)
                for i in range(len(points)):
                    line = get_line(p, points[i])
                    line_counts_with_p[line] += 1
                
                # If any line has count >= 3, then p + those 3 points = 4 collinear.
                # Wait, line_counts_with_p counts points *other than p* on the line.
                # If count >= 3, then there are 3 points + p = 4 points. Invalid.
                
                if any(c >= 3 for c in line_counts_with_p.values()):
                    continue
                
                # If valid, calculate new triples
                # New triples are those involving p.
                # A triple (p, i, j) is new if i, j are in points and collinear with p.
                # We already know p is not on any line with 3 existing points.
                # So max 2 existing points on any line with p.
                # So p can form at most C(2,2)=1 triple? No.
                # If p is on a line with 2 existing points i, j, then (p, i, j) is a triple.
                # Can p be on multiple such lines? No, p is a single point.
                # So p can be part of at most 1 triple?
                # Wait. If p is on line L1 with i, j. And on line L2 with k, l.
                # Then (p, i, j) and (p, k, l) are two triples.
                # So p can be in multiple triples.
                
                # Count triples involving p:
                # For each line passing through p and an existing point, count how many other existing points are on that line.
                # If count is 2, it forms 1 triple.
                # If count is 1, it forms 0 triples (just a pair).
                
                # We already have line_counts_with_p.
                # The value is the number of existing points on that line.
                # If value is 2, it contributes 1 triple.
                # If value is 1, it contributes 0.
                
                new_triples = 0
                for c in line_counts_with_p.values():
                    if c == 2:
                        new_triples += 1
                    elif c > 2:
                        # Should have been caught by validity check
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if new_triples > best_new_triples:
                    best_new_triples = new_triples
                    best_candidate = p
                    # If we find a candidate that creates 2 or more triples, it's very good.
                    # But we should check all to be sure?
                    # For speed, we can break if we find a "perfect" candidate?
                    # No, let's just pick the best among the 50.
            
            if best_candidate is None:
                # No valid candidate found in this sample.
                # This means we are stuck.
                # Backtracking is complex here.
                # Instead, let's try a random valid point from a larger sample or restart trial.
                # For simplicity, if stuck, we break and restart the trial.
                break
                
            points.append(best_candidate)
            used_points.add(best_candidate)
        
        if len(points) == 20:
            score = calc_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
            
    if best_points is None:
        # Fallback
        best_points = [[x, x * x] for x in range(-10, 10)]
        
    return best_points
