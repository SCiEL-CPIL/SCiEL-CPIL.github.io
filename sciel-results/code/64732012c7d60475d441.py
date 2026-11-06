import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute collinear triples structure
        # collinear_with[i][j] is a bitmask of k such that i, j, k are collinear
        collinear_with = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # Find all k such that i, j, k are collinear
                mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        mask |= (1 << k)
                collinear_with[i][j] = mask
                collinear_with[j][i] = mask

        best_score = 0
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
                
            # Pruning
            if current_size + (n - step) <= best_score:
                return
                
            idx = step
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            # Check if adding idx creates a triple with any two points in current_mask
            valid = True
            conflict_mask = 0
            # Iterate through all j in current_mask
            # We can iterate through bits in current_mask
            temp_mask = current_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                j = (lsb.bit_length() - 1)
                temp_mask ^= lsb
                
                # Check if any k in current_mask is collinear with idx and j
                # collinear_with[idx][j] contains all such k
                if collinear_with[idx][j] & current_mask:
                    valid = False
                    break
                    
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def greedy_construct():
        """
        Construct a set of 20 points by starting with a 4x5 grid and perturbing points
        to break 4-collinear sets.
        """
        # Base grid 4x5
        base_pts = []
        for y in range(4):
            for x in range(5):
                # Center it roughly
                base_pts.append((x - 2, y - 1))
        
        pts = base_pts[:]
        
        # Iteratively fix 4-collinear sets
        for _ in range(100):
            if not has_4_collinear(pts):
                break
            
            # Find a line with 4+ points
            fixed = False
            seen_lines = set()
            for i in range(20):
                for j in range(i + 1, 20):
                    key = get_line_key(pts[i], pts[j])
                    if key is None or key in seen_lines:
                        continue
                    seen_lines.add(key)
                    A, B, C = key
                    line_pts = []
                    for k in range(20):
                        x3, y3 = pts[k]
                        if A * x3 + B * y3 + C == 0:
                            line_pts.append(k)
                    if len(line_pts) >= 4:
                        # Pick a random point from this line to move
                        idx_to_move = random.choice(line_pts)
                        old_p = pts[idx_to_move]
                        
                        # Try to move to a nearby random location
                        moved = False
                        for _ in range(50):
                            new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                            if new_p in pts:
                                continue
                            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                                continue
                            
                            # Check if this move creates a new 4-collinear set
                            # Simple check: just try it
                            new_pts = pts[:]
                            new_pts[idx_to_move] = new_p
                            if not has_4_collinear(new_pts):
                                pts = new_pts
                                moved = True
                                break
                        if not moved:
                            # If we can't fix it locally, try a bigger jump
                            for _ in range(50):
                                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                                if new_p in pts:
                                    continue
                                new_pts = pts[:]
                                new_pts[idx_to_move] = new_p
                                if not has_4_collinear(new_pts):
                                    pts = new_pts
                                    moved = True
                                    break
                        if moved:
                            fixed = True
                            break
                if fixed:
                    break
            if not fixed:
                break
                
        return pts

    def local_search(pts, max_iterations=2000):
        current_pts = [tuple(p) for p in pts]
        if has_4_collinear(current_pts):
            # Repair
            for _ in range(100):
                if not has_4_collinear(current_pts):
                    break
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in current_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    current_pts[idx] = new_p
            if has_4_collinear(current_pts):
                return None, float('inf')

        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        
        for i in range(max_iterations):
            if elapsed() > deadline - 10:
                break
                
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Try local moves
            found_improvement = False
            for _ in range(20):
                new_p = (old_p[0] + random.randint(-3, 3), old_p[1] + random.randint(-3, 3))
                if new_p in current_pts:
                    continue
                if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                    continue
                    
                new_pts = current_pts[:]
                new_pts[idx] = new_p
                
                if has_4_collinear(new_pts):
                    continue
                    
                new_score = calculate_score(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts
                    current_pts = new_pts
                    current_score = new_score
                    found_improvement = True
                    break
                elif new_score == current_score and random.random() < 0.1:
                    current_pts = new_pts
                    current_score = new_score
                    
            if not found_improvement and random.random() < 0.05:
                # Random restart of the move
                idx = random.randint(0, 19)
                current_pts[idx] = (random.randint(-100, 100), random.randint(-100, 100))
                if has_4_collinear(current_pts):
                    # Revert if invalid
                    current_pts[idx] = old_p
                else:
                    current_score = calculate_score(current_pts)
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = current_pts[:]
                        
        return best_pts, best_score

    best_points = None
    best_val = float('inf')
    
    # Generate initial candidates using greedy construction
    candidates = []
    for _ in range(10):
        if elapsed() > deadline - 30:
            break
        cand = greedy_construct()
        if not has_4_collinear(cand):
            candidates.append(cand)
            
    # Also add some random ones
    for _ in range(5):
        if elapsed() > deadline - 30:
            break
        pts = []
        while len(pts) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in pts:
                pts.append(p)
        if not has_4_collinear(pts):
            candidates.append(pts)
            
    random.shuffle(candidates)
    
    for cand in candidates:
        if elapsed() > deadline - 20:
            break
        
        pts, score = local_search(cand, max_iterations=500)
        
        if pts is not None and score < best_val:
            best_val = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        best_points = greedy_construct()
        if has_4_collinear(best_points):
             # Fix
             for _ in range(100):
                 if not has_4_collinear(best_points):
                     break
                 idx = random.randint(0, 19)
                 old_p = best_points[idx]
                 new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                 if new_p not in best_points and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                     best_points[idx] = new_p

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
