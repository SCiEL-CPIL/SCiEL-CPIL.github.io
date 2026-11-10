"""
Minimize largest 3-collinear-free subset for 20 points via combinatorial hypergraph search.

Strategy:
Instead of searching the space of 20 point coordinates (huge, continuous-like), 
we search the space of possible "collinear triple" structures.
A valid configuration corresponds to a set of triples T (where each triple is a set of 3 indices) 
such that:
1. No two triples share 2 indices (otherwise 4 points would be collinear if they are on the same line, 
   or the geometry is inconsistent. Actually, if (a,b,c) and (a,b,d) are triples, then a,b,c,d are collinear. 
   So no two triples can share 2 indices.)
2. The structure must be geometrically realizable in Z^2.

We use a heuristic search over the set of triples.
Objective: Maximize the number of triples, or more precisely, minimize the independence number of the 3-uniform hypergraph defined by the triples.
Minimizing the independence number is equivalent to making the hypergraph "dense" in a way that covers vertices with many constraints.
Actually, the score is the size of the largest independent set.
If we have a fixed number of points N=20, maximizing the number of triples generally reduces the max independent set, 
but the structure matters. A Steiner triple system (if it existed for N=20, which it doesn't exactly, but close) 
would have each pair in at most one triple.

We will:
1. Generate a candidate set of triples.
2. Check if they are geometrically realizable (linear dependencies).
3. If realizable, compute the score.
4. Use Simulated Annealing to modify the triple set (add/remove triples) to minimize the score.

Geometric Realizability Check:
Given a set of triples, we need to assign coordinates to 20 points such that for every triple (i,j,k), 
the points are collinear.
This is a system of linear equations.
Let p_i = (x_i, y_i).
Collinearity of i, j, k means det([p_i, p_j, p_k]) = 0.
This is non-linear in coordinates.
However, we can fix 3 points to break symmetry (e.g., (0,0), (1,0), (0,1)) and try to solve for the rest?
Or better: We can use the fact that we just need *some* integer realization.
A simpler check: If the triple structure is "too dense", it might not be realizable.
But for N=20, we can try to construct the points from the triples.
Actually, it's easier to go the other way:
The current program generates points and checks triples.
This new program generates triples and checks if points exist.

Wait, generating triples and finding points is hard.
Alternative: 
Use a known structure: The affine plane AG(4,2) has 16 points. 
The affine plane AG(5,2) has 25 points.
We need 20 points.
Consider the "Pappus configuration" or other small configurations.

Let's try a different angle:
The score is the independence number.
For a 3-uniform hypergraph, the independence number is the max set of vertices with no edge fully contained in it.
We want to MINIMIZE this.
This is equivalent to finding a hypergraph on 20 vertices, realizable in Z^2, with minimum independence number.

Known bounds:
For N=20, the maximum number of points in general position is 20 (score 20).
We want to lower it.
A random set of 20 points in [100,100]^2 has very few collinear triples, so score is close to 20.
To lower the score, we need MANY collinear triples.
The maximum number of collinear triples in a set of N points with no 4 collinear is bounded.
If we can pack as many triples as possible, we force the independence number down.

Strategy:
1. Start with a random set of 20 points.
2. Identify all collinear triples.
3. This defines a hypergraph H.
4. Compute alpha(H) (independence number).
5. Local search: Move a point to create NEW triples or destroy existing ones?
   We want to INCREASE the number of triples to decrease alpha.
   Moving a point onto a line defined by 2 other points creates a new triple.
   But we must ensure no 4 collinear.
   So, we look for "almost collinear" situations or move points to lie on lines with exactly 2 other points.

Let's implement a "Triple-Maximizing" Local Search.
1. Generate random valid points.
2. Count triples.
3. For each point, try moving it to a position that lies on the line defined by a pair of other points, 
   such that the new line has exactly 3 points (the moved one + the pair).
   This adds 1 triple.
   Check if it creates a 4-collinear (it shouldn't if the line had only 2 points before).
   Check if it creates other unintended triples (it might).
   If the move increases the total number of triples, accept it.
   This greedy approach should find a configuration with many triples.
4. Once we have a high-triple configuration, use Simulated Annealing to fine-tune.
   The objective function for SA should be the independence number (computed exactly), 
   but since exact computation is expensive, we can use the number of triples as a proxy 
   or a mix. However, the task requires minimizing the score (independence number).
   The number of triples is a good proxy for "density", which correlates with lower independence number.
   But two hypergraphs with the same number of triples can have different independence numbers.
   We will use the exact independence number as the objective for the final stages, 
   and triple count for the initial greedy phase.

Exact Independence Number Calculation:
Use the branch and bound from the current program, but optimized.
For N=20, it's fast enough if called occasionally.

Algorithm:
1. Init: Random points, no 4 collinear.
2. Greedy Phase:
   For each point i:
      For each pair (j, k) not involving i:
         Calculate line L(j, k).
         Find an integer point p on L(j, k) such that:
            - p is not in the current set.
            - No other point is on L(j, k) (to avoid 4-collinear).
            - p is within bounds.
            - Moving i to p doesn't create 4-collinear with other pairs.
         If such p exists, and moving i to p increases the total triple count, do it.
   Repeat until no improvement.
3. SA Phase:
   Objective: Minimize alpha(H).
   Moves: 
      a) Move a point to a random valid location.
      b) "Swap" two points? No, just move.
   Accept if alpha decreases, or with prob exp(-delta/T).
   To speed up, we can estimate alpha or use triple count as a surrogate for most steps, 
   and verify with exact alpha every K steps.
   
   Actually, exact alpha for N=20 is very fast (< 1ms in C, maybe 10-50ms in Python).
   Let's benchmark. 2^20 states is too many, but branch and bound prunes heavily.
   The current program's `_max_independent_set` is O(N^3) per node? No, it's recursive.
   Let's assume it takes ~10ms.
   In 120s, we can do ~12,000 evaluations.
   This is enough for a decent SA.

Let's refine the Greedy Phase to be more effective.
Instead of just adding triples, we should look for moves that "cluster" points.
A line with 3 points is good.
Can we have two lines sharing a point? Yes, e.g., (0,0)-(1,1)-(2,2) and (0,0)-(1,0)-(2,0).
This creates two triples sharing (0,0).
This structure reduces the independence number more than disjoint triples?
Probably.

Let's implement the "Move to Line" heuristic carefully.
When moving point i to line L(j, k):
1. Check if L(j, k) has any other points. If yes, skip (would make 4 collinear).
2. Check if the new point p creates any 4-collinear with other points.
   i.e., for any other pair (u, v), does p lie on L(u, v) and does L(u, v) have another point?
   If so, skip.
3. If valid, calculate new triple count.
   Old triples involving i are removed.
   New triples involving p are added.
   Specifically, p forms a triple with (j, k).
   Does p form triples with other pairs?
   For any other point m, does p, m, and some other point n form a line?
   We need to check all pairs (m, n) to see if p is on L(m, n).
   If yes, and L(m, n) has no other points, it's a new triple.
   
This is O(N^2) per move candidate.
There are O(N^2) pairs (j, k) for each i.
Total O(N^4) per full pass. N=20, 160,000 ops. Very fast.

We will run this greedy phase until convergence.
Then SA.

One detail: The problem asks for integer points with |x|, |y| <= 100.
When finding p on L(j, k), we need to find an integer point.
Line through (x1, y1) and (x2, y2).
dx = x2 - x1, dy = y2 - y1.
g = gcd(abs(dx), abs(dy)).
The points on the line are (x1 + t*dx/g, y1 + t*dy/g) for integer t.
We need to find t such that the point is in bounds and not already in the set.
Also, we need to ensure that the segment or line doesn't have other points.
We already checked that L(j, k) has no other points in the current set.
But we must ensure that the *new* point p doesn't lie on a line with two other points u, v 
such that L(u, v) has a fourth point? No, just that L(u, v) has no *other* points besides u, v, p.
Wait, if p is on L(u, v), and u, v are in the set, then {u, v, p} is a triple.
If there is another point w on L(u, v), then {u, v, p, w} are 4 collinear. Invalid.
So the check "L(j, k) has no other points" is necessary for the line we are moving TO.
But we also need to check if p creates a 4-collinear on ANY other line.
So for every other pair (u, v), check if p is on L(u, v).
If yes, check if L(u, v) has any other point w (besides u, v, p).
If yes, invalid.

This check is O(N^2) per candidate p.
Since we might have multiple candidate t values for p, we should pick one that minimizes risk?
Or just pick a random valid t.

Let's code this up.
"""

import time
import random
import math
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _count_triples(points):
    """Count total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
    return count

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    """
    n = len(points)
    
    # Precompute collinear triples
    # collinear_with[i][j] = set of k such that i, j, k are collinear
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)

    best = 0
    
    def backtrack(current_set, candidates):
        nonlocal best
        if len(current_set) + len(candidates) <= best:
            return
            
        if not candidates:
            if len(current_set) > best:
                best = len(current_set)
            return

        p_idx = candidates[0]
        rest_candidates = candidates[1:]
        
        # Branch 1: Include p_idx
        # New candidates: those in rest_candidates that are not collinear with p_idx and any point in current_set
        new_candidates = []
        for k in rest_candidates:
            can_add = True
            for c in current_set:
                if k in collinear_with[p_idx][c]:
                    can_add = False
                    break
            if can_add:
                new_candidates.append(k)
        
        backtrack(current_set + [p_idx], new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_set, rest_candidates)

    backtrack([], list(range(n)))
    return best

def _find_integer_points_on_line(p1, p2, exclude_points, bounds=100):
    """
    Find integer points on the line passing through p1 and p2,
    excluding points already in exclude_points, within bounds.
    Returns a list of candidate points.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return []
        
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return []
        
    step_x = dx // g
    step_y = dy // g
    
    candidates = []
    
    # Determine range of t such that (x1 + t*step_x, y1 + t*step_y) is in bounds
    # -bounds <= x1 + t*step_x <= bounds
    # -bounds <= y1 + t*step_y <= bounds
    
    t_min = -1000
    t_max = 1000
    
    # Refine t range
    if step_x != 0:
        t_min_x = math.ceil((-bounds - x1) / step_x) if step_x > 0 else math.floor((-bounds - x1) / step_x)
        t_max_x = math.floor((bounds - x1) / step_x) if step_x > 0 else math.ceil((bounds - x1) / step_x)
        t_min = max(t_min, t_min_x)
        t_max = min(t_max, t_max_x)
    if step_y != 0:
        t_min_y = math.ceil((-bounds - y1) / step_y) if step_y > 0 else math.floor((-bounds - y1) / step_y)
        t_max_y = math.floor((bounds - y1) / step_y) if step_y > 0 else math.ceil((bounds - y1) / step_y)
        t_min = max(t_min, t_min_y)
        t_max = min(t_max, t_max_y)
        
    for t in range(t_min, t_max + 1):
        px = x1 + t * step_x
        py = y1 + t * step_y
        p = [px, py]
        if p not in exclude_points:
            candidates.append(p)
            
    return candidates

def _is_move_valid(points, idx, new_p):
    """
    Check if moving points[idx] to new_p results in a valid configuration (no 4 collinear).
    Assumes new_p is not already in points (except at idx).
    """
    # Temporarily replace
    old_p = points[idx]
    points[idx] = new_p
    
    # Check no 4 collinear
    valid = _check_no_four_collinear(points)
    
    # Revert
    points[idx] = old_p
    
    return valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 20
    bounds = 100
    
    # 1. Generate initial random valid configuration
    def generate_initial():
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 10000:
            attempts += 1
            p = [random.randint(-bounds, bounds), random.randint(-bounds, bounds)]
            if p in pts:
                continue
            # Check if adding p creates 4-collinear
            valid = True
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx * (p[1] - y1) == dy * (p[0] - x1):
                        # p is on line ij
                        count = 2
                        for k in range(len(pts)):
                            if k == i or k == j:
                                continue
                            x3, y3 = pts[k]
                            if dx * (y3 - y1) == dy * (x3 - x1):
                                count += 1
                        if count >= 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                pts.append(p)
        if len(pts) < n:
            # Fallback: use a grid-like pattern with perturbations
            pts = []
            for i in range(n):
                x = (i % 5) * 20 - 40
                y = (i // 5) * 20 - 40
                pts.append([x + random.randint(-2, 2), y + random.randint(-2, 2)])
            # Ensure distinct
            seen = set()
            final_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    final_pts.append(p)
            while len(final_pts) < n:
                p = [random.randint(-bounds, bounds), random.randint(-bounds, bounds)]
                if p not in seen:
                    seen.add(p)
                    final_pts.append(p)
            return final_pts
        return pts

    current_points = generate_initial()
    
    # Ensure validity (generate_initial should guarantee it, but just in case)
    if not _check_no_four_collinear(current_points):
        # Force fix
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = current_points[i], current_points[j], current_points[k], current_points[l]
                        # Check collinear
                        if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]) and \
                           (p2[0]-p1[0])*(p4[1]-p1[1]) == (p4[0]-p1[0])*(p2[1]-p1[1]):
                            # Move p4
                            while True:
                                new_p = [random.randint(-bounds, bounds), random.randint(-bounds, bounds)]
                                if new_p in current_points:
                                    continue
                                current_points[l] = new_p
                                if _check_no_four_collinear(current_points):
                                    break
        # Re-check
        if not _check_no_four_collinear(current_points):
             # Last resort: random restart
             current_points = generate_initial()

    current_triples = _count_triples(current_points)
    current_score = _max_independent_set(current_points)
    
    best_points = current_points[:]
    best_score = current_score
    best_triples = current_triples
    
    # 2. Greedy Phase: Maximize triples
    # Try to move points to lie on lines defined by pairs of other points
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        for idx in range(n):
            if time.time() > deadline - 20.0:
                break
            best_move_for_idx = None
            best_triple_gain = 0
            
            # Try moving idx to a line defined by (j, k)
            for j in range(n):
                if j == idx: continue
                for k in range(j+1, n):
                    if k == idx: continue
                    
                    p_j = current_points[j]
                    p_k = current_points[k]
                    
                    # Check if line (j, k) has any other points
                    line_has_other = False
                    for m in range(n):
                        if m == j or m == k or m == idx:
                            continue
                        p_m = current_points[m]
                        dx = p_k[0] - p_j[0]
                        dy = p_k[1] - p_j[1]
                        if dx * (p_m[1] - p_j[1]) == dy * (p_m[0] - p_j[0]):
                            line_has_other = True
                            break
                    if line_has_other:
                        continue
                        
                    # Find candidate points on this line
                    exclude = set(map(tuple, current_points))
                    candidates = _find_integer_points_on_line(p_j, p_k, exclude, bounds)
                    
                    if not candidates:
                        continue
                        
                    # Pick a random candidate (or the one closest to current position?)
                    # Let's try a few random candidates
                    random.shuffle(candidates)
                    for new_p in candidates[:5]: # Try up to 5 candidates
                        if not _is_move_valid(current_points, idx, new_p):
                            continue
                        
                        # Calculate gain in triples
                        # Old triples involving idx are lost
                        # New triples involving new_p are gained
                        # This is complex to calculate exactly without full recount.
                        # Let's just do the move and recount if it seems promising?
                        # Or approximate: 
                        # The move adds at least 1 triple (with j, k).
                        # It might add more.
                        # It removes triples involving idx.
                        
                        # Let's just tentatively apply and count
                        old_p = current_points[idx]
                        current_points[idx] = new_p
                        new_triples = _count_triples(current_points)
                        gain = new_triples - current_triples
                        
                        if gain > best_triple_gain:
                            best_triple_gain = gain
                            best_move_for_idx = new_p
                            # Keep the point moved for now? No, revert for next candidate
                            # We will apply the best move after the loop
                        current_points[idx] = old_p
                        
            if best_move_for_idx is not None:
                # Apply the best move
                current_points[idx] = best_move_for_idx
                current_triples += best_triple_gain
                improved = True
                # Update score occasionally?
                if random.random() < 0.1:
                    new_score = _max_independent_set(current_points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = current_points[:]
    
    # 3. Simulated Annealing Phase
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    # Use score as objective, but triple count as a filter to avoid bad moves
    while time.time() < deadline and T > T_min:
        idx = random.randint(0, n-1)
        old_p = current_points[idx]
        
        # Generate a new point
        # Bias towards points that might create triples?
        # Just random for now
        new_p = [random.randint(-bounds, bounds), random.randint(-bounds, bounds)]
        
        if new_p == old_p or new_p in current_points:
            continue
            
        if not _is_move_valid(current_points, idx, new_p):
            continue
            
        current_points[idx] = new_p
        new_triples = _count_triples(current_points)
        
        # Heuristic: If triples decreased significantly, reject immediately?
        # Or just use SA on score.
        
        # Calculate new score (expensive)
        # To save time, only calculate exact score if triples increased or randomly
        if new_triples > current_triples or random.random() < 0.2:
            new_score = _max_independent_set(current_points)
        else:
            # Estimate: assume score stays same or slightly worse
            # If we reject, we revert
            # If we accept, we need exact score later?
            # Let's just calculate exact score always for simplicity, 
            # but limit the number of SA steps.
            new_score = _max_independent_set(current_points)
            
        current_points[idx] = old_p # Revert for decision
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_score / T)
            
        if accept:
            current_points[idx] = new_p
            current_triples = new_triples
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
                
        T *= alpha

    # Final check
    if not _check_no_four_collinear(best_points):
        # Should not happen, but just in case
        return generate_initial()
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(f"Triples: {_count_triples(pts)}")

