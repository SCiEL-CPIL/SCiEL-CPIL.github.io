import random
import time
import itertools

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _has_four_collinear(points):
    n = len(points)
    # Check all combinations of 4
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_collinear(points[i], points[j], points[k]) and _are_collinear(points[i], points[j], points[l]):
                        return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    # Build list of collinear triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    # Branch and bound for Maximum Independent Set in hypergraph
    # A subset S is independent if it contains no triple from `triples`.
    # This is equivalent to: for every triple (i,j,k), S cannot contain all i, j, k.
    # Or: if S contains i and j, it cannot contain k.
    
    # Use a simple recursive search with pruning
    best = 0
    
    # Heuristic: Order points by degree (number of triples they are in)
    # Points in more triples are "more constrained".
    
    def dfs(idx, current_size, used):
        nonlocal best
        if current_size > best:
            best = current_size
            
        if idx == n:
            return
            
        # Pruning: if current_size + (n - idx) <= best, return
        if current_size + (n - idx) <= best:
            return
            
        # Option 1: Include point idx
        # Check if including idx violates any triple with already included points
        valid = True
        for t in triples:
            if idx in t:
                # Check if the other two are in 'used'
                others = [x for x in t if x != idx]
                if others[0] in used and others[1] in used:
                    valid = False
                    break
        if valid:
            dfs(idx + 1, current_size + 1, used | {idx})
            
        # Option 2: Exclude point idx
        dfs(idx + 1, current_size, used)

    # Optimization: We can use a bitmask for 'used' if n <= 20
    # n=20 fits in an int.
    
    def dfs_mask(idx, current_size, mask):
        nonlocal best
        if current_size > best:
            best = current_size
        if idx == n:
            return
        if current_size + (n - idx) <= best:
            return
        
        # Include idx
        bit = 1 << idx
        valid = True
        for t in triples:
            if idx in t:
                others = [x for x in t if x != idx]
                if (mask >> others[0]) & 1 and (mask >> others[1]) & 1:
                    valid = False
                    break
        if valid:
            dfs_mask(idx + 1, current_size + 1, mask | bit)
            
        # Exclude idx
        dfs_mask(idx + 1, current_size, mask)

    dfs_mask(0, 0, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial random valid configuration
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    points = []
    while len(points) < 20:
        p = random_point()
        if p not in points:
            # Check if adding p creates 4-collinear with existing points
            valid = True
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if _are_collinear(points[i], points[j], p):
                        # Check if there's a 4th point on this line
                        # Actually, just check if any 3 existing points are collinear with p
                        # Wait, if p is collinear with points[i] and points[j], we need to ensure
                        # that no other point points[k] is also on that line.
                        # So check all k
                        for k in range(j+1, len(points)):
                            if _are_collinear(points[i], points[j], points[k]):
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            if valid:
                points.append(p)
        
    if len(points) < 20:
        # Fallback: simple grid with perturbation if random fails (unlikely)
        points = [[x, y] for x in range(-10, 10)] # Parabola, score 20
        
    best_points = points
    best_score = _max_no_three_collinear(points)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        # Strategy: Try to move to a line defined by two other points
        # to create a new 3-collinearity
        found_move = False
        # Shuffle other points to pick random pairs
        other_indices = [i for i in range(20) if i != idx]
        random.shuffle(other_indices)
        
        for a in range(len(other_indices)):
            i1 = other_indices[a]
            for b in range(a+1, len(other_indices)):
                i2 = other_indices[b]
                p1 = points[i1]
                p2 = points[i2]
                
                # Check if line p1-p2 has any other points
                # If it does, moving to it might create 4-collinear
                # We want to move to a line with EXACTLY 2 points (p1, p2)
                # So that adding the 3rd makes it 3.
                line_has_other = False
                for k in range(20):
                    if k == idx or k == i1 or k == i2:
                        if _are_collinear(p1, p2, points[k]):
                            line_has_other = True
                            break
                if line_has_other:
                    continue
                
                # Find a new point on line p1-p2
                # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
                # We need integer solutions.
                # Simple way: parametrize
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Try a few steps from p1
                for t in range(-5, 6):
                    if t == 0: continue # p1
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    new_p = [nx, ny]
                    if new_p == old_p:
                        continue
                    if new_p in points:
                        continue
                    
                    # Check if new_p creates 4-collinearities
                    # It is on line p1-p2 (3 points: p1, p2, new_p).
                    # Check if new_p is on any OTHER line with 3 existing points.
                    valid_move = True
                    for k in range(20):
                        if k == idx or k == i1 or k == i2:
                            continue
                        # Check if new_p, points[k], and some other point form a line with a 4th?
                        # Actually, we just need to ensure no line has 4 points.
                        # The only new lines introduced are those passing through new_p.
                        # We already checked line p1-p2 has only p1, p2.
                        # So we need to check if new_p is collinear with any pair (k, l)
                        # such that there is a 4th point?
                        # No, if new_p is collinear with k and l, the line has new_p, k, l.
                        # Is there a 4th?
                        # We must check if any line through new_p has >= 4 points.
                        # This is expensive.
                        # Heuristic: Just check if new_p is collinear with any 3 existing points.
                        # If it is, that line has 4 points (new_p + 3 existing).
                        # If it is collinear with 2 existing points, that line has 3 points.
                        # If it is collinear with 1 existing point, that line has 2 points.
                        
                        # So: Check if there exist 3 distinct existing points A, B, C
                        # such that new_p, A, B, C are collinear.
                        # This is equivalent to: new_p is on a line that contains 3 existing points.
                        
                        # We can precompute lines with 3+ points?
                        # Or just check pairs.
                        # If new_p is collinear with A and B, check if any C is also collinear with A and B.
                        
                        for m in range(20):
                            if m == k or m == i1 or m == i2:
                                continue
                            if _are_collinear(new_p, points[k], points[m]):
                                # Check if there's a 4th point on this line
                                # We know new_p, k, m are on the line.
                                # Check if any other point n is on this line.
                                for n in range(20):
                                    if n == idx or n == k or n == m or n == i1 or n == i2:
                                        continue
                                    if _are_collinear(points[k], points[m], points[n]):
                                        valid_move = False
                                        break
                                if not valid_move:
                                    break
                        if not valid_move:
                            break
                    if valid_move:
                        # Perform move
                        points[idx] = new_p
                        found_move = True
                        # Calculate new score
                        new_score = _max_no_three_collinear(points)
                        if new_score < best_score:
                            best_score = new_score
                            best_points = points.copy()
                        # If new_score is not better, we might revert or keep?
                        # Keeping might allow further moves.
                        # But if it's worse, maybe revert?
                        # Let's keep it for now (simulated annealing style)
                        break
            if found_move:
                break
        
        if not found_move:
            # Random move
            new_p = random_point()
            if new_p not in points:
                # Check validity
                # (Simplified: just check 4-collinear)
                points[idx] = new_p
                if not _has_four_collinear(points):
                    new_score = _max_no_three_collinear(points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = points.copy()
                else:
                    points[idx] = old_p

    return best_points
