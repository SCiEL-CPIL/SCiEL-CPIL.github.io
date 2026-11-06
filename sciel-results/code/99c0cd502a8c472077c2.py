"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138  # Safety margin
    
    def collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])
    
    # Precompute line signatures for pairs
    def line_sig(p1, p2):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        if dx == 0:
            return (0, 0, p1[0])  # vertical
        if dy == 0:
            return (0, 1, p1[1])  # horizontal
        g = abs(dy)
        if g < abs(dx):
            g = abs(dx)
        # Normalize
        import math
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        # Canonical form: dx_n > 0 or (dx_n == 0 and dy_n > 0)
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        # Line: dy_n*x - dx_n*y = c
        c = dy_n*p1[0] - dx_n*p1[1]
        return (dx_n, dy_n, c)
    
    def build_line_map(points):
        """Map from line signature to list of point indices on that line"""
        line_map = defaultdict(list)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                sig = line_sig(points[i], points[j])
                line_map[sig].append(i)
                line_map[sig].append(j)
        # Deduplicate
        for sig in line_map:
            line_map[sig] = list(set(line_map[sig]))
        return line_map
    
    def count_triples_from_linemap(line_map):
        count = 0
        for sig, pts in line_map.items():
            if len(pts) >= 3:
                from itertools import combinations
                count += len(list(combinations(pts, 3)))
        return count
    
    def check_no_four_from_linemap(line_map):
        for sig, pts in line_map.items():
            if len(pts) >= 4:
                return False
        return True
    
    def max_no_three_collinear(points):
        n = len(points)
        # Build set of collinear triples
        triple_set = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i], points[j], points[k]):
                        triple_set.add((i, j, k))
        
        # For branch and bound, we need efficient conflict checking
        # Precompute: for each pair (i,j), the set of k such that (i,j,k) is collinear
        pair_conflicts = defaultdict(set)
        for (i, j, k) in triple_set:
            pair_conflicts[(i,j)].add(k)
            pair_conflicts[(j,i)].add(k)
        
        best = [0]
        
        def bnb(idx, current_size, chosen):
            if idx >= n:
                best[0] = max(best[0], current_size)
                return
            
            if current_size + (n - idx) <= best[0]:
                return
            
            # Skip idx
            bnb(idx+1, current_size, chosen)
            
            # Try to include idx
            valid = True
            for a in chosen:
                for b in chosen:
                    if a == b:
                        continue
                    if a < b:
                        if b in pair_conflicts.get((a, idx), set()):
                            valid = False
                            break
                        if a in pair_conflicts.get((idx, b), set()):
                            valid = False
                            break
                    else:
                        if a in pair_conflicts.get((idx, b), set()):
                            valid = False
                            break
                        if b in pair_conflicts.get((idx, a), set()):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            # Simpler check
            valid = True
            chosen_list = sorted(chosen)
            for ai in range(len(chosen_list)):
                for bi in range(ai+1, len(chosen_list)):
                    a = chosen_list[ai]
                    b = chosen_list[bi]
                    # Check if adding idx creates a collinear triple with a and b
                    # Need to check if (a,b,idx) is in triple_set
                    if a < b < idx:
                        if (a, b, idx) in triple_set:
                            valid = False
                            break
                    elif a < idx < b:
                        if (a, idx, b) in triple_set:
                            valid = False
                            break
                    elif idx < a < b:
                        if (idx, a, b) in triple_set:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                chosen.add(idx)
                bnb(idx+1, current_size+1, chosen)
                chosen.remove(idx)
        
        bnb(0, 0, set())
        return best[0]
    
    # Initial: parabola
    points = [[x, x*x] for x in range(-10, 10)]
    
    # Candidate pool - use a smaller set for speed
    candidates = []
    for x in range(-50, 51):
        for y in range(-50, 51):
            candidates.append([x, y])
    random.seed(42)
    random.shuffle(candidates)
    
    best_points = points[:]
    best_score = 20
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        line_map = build_line_map(points)
        cur_triples = count_triples_from_linemap(line_map)
        
        for i in range(20):
            if time.time() > deadline - 10:
                break
            old_point = points[i]
            for cand in candidates[:200]:
                if time.time() > deadline - 10:
                    break
                if cand == old_point:
                    continue
                if cand in points:
                    continue
                
                new_points = points[:]
                new_points[i] = cand
                new_line_map = build_line_map(new_points)
                if not check_no_four_from_linemap(new_line_map):
                    continue
                
                new_triples = count_triples_from_linemap(new_line_map)
                if new_triples > cur_triples:
                    points = new_points
                    cur_triples = new_triples
                    improved = True
                    break
    
    # Compute score if time permits
    if time.time() < deadline - 5:
        score = max_no_three_collinear(points)
        if score < best_score:
            best_points = points[:]
            best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
