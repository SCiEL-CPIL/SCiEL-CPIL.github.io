"""Rewritten program: Simulated Annealing to maximize collinear triples (minimizing max no-3-collinear subset)."""
import random
import itertools
import time
import math
from collections import defaultdict

def _no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    
    def line_rep(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        if A == 0 and B == 0:
            return None
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0) else 1
        if g == 0:
            g = 1
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    line_to_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            rep = line_rep(i, j)
            if rep is None:
                continue
            line_to_points[rep].add(i)
            line_to_points[rep].add(j)
    
    for rep, pts in line_to_points.items():
        if len(pts) >= 4:
            return False
    return True

def _count_collinear_triples(points):
    """Count the number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    count += 1
    return count

def _largest_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    
    # Precompute conflict masks: 
    # conflict[i] is a bitmask of points j such that there exists k where (i,j,k) is collinear?
    # No, that's not quite right for the B&B logic.
    # Let's use the logic: 
    # If we select i, we cannot select any j,k that form a triple with i and each other?
    # Actually, if we select i, and we have already selected j, we cannot select k if (i,j,k) is collinear.
    
    # Let's precompute: for each pair (i,j), the set of k such that (i,j,k) is collinear.
    # triple_mask[i][j] = bitmask of k's.
    triple_mask = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    mask |= (1 << k)
            triple_mask[i][j] = mask
            triple_mask[j][i] = mask

    best_size = 0
    
    def backtrack(selected_mask, candidates_mask):
        nonlocal best_size
        # Upper bound: current selected count + popcount(candidates)
        current_count = bin(selected_mask).count('1')
        if current_count + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_count > best_size:
                best_size = current_count
            return
        
        # Pick the first available candidate
        # Heuristic: pick the one with fewest conflicts? Or just lowest index.
        # Let's pick the lowest index for simplicity, or one that conflicts most to prune?
        # Picking one that conflicts most might reduce candidates faster.
        
        # Find a candidate
        # We can iterate through candidates
        # To speed up, we can just pick the lowest set bit.
        
        # Optimization: If we pick i, we must remove from candidates any j such that 
        # there exists k in selected_mask where (i,j,k) is collinear.
        # AND we must remove any j such that (i,j,k) is collinear for some k in candidates? No.
        # The condition is: the final set contains no triple.
        # If I add i to selected, I must ensure that for any j in selected, no k in (selected U candidates) forms a triple.
        # Actually, if I add i, I must remove from candidates any j such that 
        # (i, j, k) is a triple for some k in selected.
        # Wait, if k is in selected, and I add i, and j is in candidates, if (i,j,k) is a triple, I cannot keep j.
        # So, new_candidates = candidates & ~ (union of triple_mask[i][k] for all k in selected)
        
        # Also, I must ensure i itself doesn't form a triple with two points in selected.
        # Check: for all k, m in selected, is (i,k,m) a triple?
        # If yes, we cannot add i.
        
        # Let's find the best i to add.
        # We can try all candidates.
        
        # To make it faster, we can just pick one candidate and branch?
        # No, we must try adding each candidate?
        # Standard B&B for Independent Set:
        # 1. If candidates is empty, update best.
        # 2. Pick a vertex v from candidates.
        # 3. Branch 1: Include v.
        #    - Check if v conflicts with any pair in selected.
        #    - If valid, new_selected = selected | v.
        #    - new_candidates = candidates & ~v & ~ (conflicts with v given selected).
        #    - Recurse.
        # 4. Branch 2: Exclude v.
        #    - new_selected = selected.
        #    - new_candidates = candidates & ~v.
        #    - Recurse.
        
        # This is 2^n. With n=20, it's 1M. Pruning is key.
        
        # Let's pick the first available candidate.
        # Find lowest set bit in candidates_mask
        # Use a loop or bit trick.
        
        # To improve pruning, we should pick the candidate with the highest degree of conflict?
        # Let's just pick the first one for now.
        
        # Get list of candidates
        # This is slow in Python. Let's use bit operations.
        
        # Let's try a different approach: Recursive on the set of points.
        # But we need to be fast.
        
        # Let's stick to the bitset B&B.
        
        # Find an index i in candidates_mask
        # We can't easily iterate bits in Python without a loop.
        # Let's convert to list of indices?
        # Candidates are few.
        
        cand_indices = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            cand_indices.append(idx)
            temp ^= lsb
            
        if not cand_indices:
            if current_count > best_size:
                best_size = current_count
            return
            
        # Heuristic: pick the candidate that conflicts with the most selected points?
        # Or just the first one.
        # Let's pick the first one for simplicity, but sort by "badness"?
        
        # Let's just pick the first one.
        v = cand_indices[0]
        
        # Option 1: Exclude v
        backtrack(selected_mask, candidates_mask & ~(1 << v))
        
        # Option 2: Include v
        # Check if v forms a triple with any pair in selected
        valid = True
        sel_indices = []
        temp_sel = selected_mask
        while temp_sel:
            lsb = temp_sel & -temp_sel
            idx = (lsb.bit_length() - 1)
            sel_indices.append(idx)
            temp_sel ^= lsb
            
        for a in range(len(sel_indices)):
            for b in range(a + 1, len(sel_indices)):
                if triple_mask[sel_indices[a]][sel_indices[b]] & (1 << v):
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Calculate new candidates
            # Remove v from candidates
            new_cand = candidates_mask & ~(1 << v)
            # Remove any j in new_cand such that (v, j, k) is a triple for some k in selected
            forbidden = 0
            for k in sel_indices:
                forbidden |= triple_mask[v][k]
            
            new_cand &= ~forbidden
            
            backtrack(selected_mask | (1 << v), new_cand)

    backtrack(0, (1 << n) - 1)
    return best_size

def construct():
    """Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest no-three-collinear subset."""
    
    random.seed(42)
    
    # Initialize with random points
    points = []
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p not in points:
            points.append(p)
            
    points = [list(p) for p in points]
    
    # Ensure no 4 collinear initially?
    # If not, random restarts or local fix.
    # For simplicity, we will use Simulated Annealing which can handle violations by rejecting them.
    
    start_time = time.time()
    max_time = 100
    
    current_points = points
    current_triples = _count_collinear_triples(current_points)
    if not _no_four_collinear(current_points):
         # Try to fix or restart
         # Just restart a few times
         attempts = 0
         while not _no_four_collinear(current_points) and attempts < 100:
             points = []
             while len(points) < 20:
                 x = random.randint(-100, 100)
                 y = random.randint(-100, 100)
                 p = (x, y)
                 if p not in points:
                     points.append(p)
             current_points = [list(p) for p in points]
             attempts += 1
             if time.time() - start_time > max_time:
                 break
         
         if not _no_four_collinear(current_points):
             # Fallback to a known valid config (parabola)
             current_points = [[x, x * x] for x in range(-10, 10)]
             current_triples = _count_collinear_triples(current_points)
         else:
             current_triples = _count_collinear_triples(current_points)
             
    best_points = [p[:] for p in current_points]
    best_triples = current_triples
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Generate neighbor
        i = random.randint(0, 19)
        x, y = current_points[i]
        nx = random.randint(max(-100, x - 5), min(100, x + 5))
        ny = random.randint(max(-100, y - 5), min(100, y + 5))
        
        new_points = [p[:] for p in current_points]
        new_points[i] = [nx, ny]
        
        if len(set(map(tuple, new_points))) != 20:
            continue
            
        if not _no_four_collinear(new_points):
            continue
            
        new_triples = _count_collinear_triples(new_points)
        
        delta = new_triples - current_triples # We want to maximize triples
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [p[:] for p in current_points]
                
        T *= cooling
        
    return best_points
