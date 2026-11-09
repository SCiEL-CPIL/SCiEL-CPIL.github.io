import random
import time

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _check_no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 0
            for k in range(n):
                if k != i and k != j and _are_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 2:
                        return False
    return True

def _max_no_three_collinear(points):
    n = len(points)
    # conflict_mask[i] = bitmask of points that form a collinear triple with i and some other already-included point
    # Better approach: for each pair (i,j) that are collinear with some k, record the conflict
    # Precompute: for each point i, which pairs (a,b) with a<b, a!=i, b!=i cause a triple if i is added
    # Actually, let's use a different approach:
    # conflict[i][j] = set of k such that i,j,k are collinear (k != i, k != j)
    # When adding point i, check if any two already-included points j,k have i on line jk
    
    # For efficiency: precompute for each point i, a list of (j, k) pairs where i,j,k are collinear
    # Then when considering adding i, check if both j and k are in the included set
    
    triple_partners = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triple_partners[i].append((j, k))
                    triple_partners[j].append((i, k))
                    triple_partners[k].append((i, j))
    
    # Sort points by number of conflicts (most constrained first) for better pruning
    order = sorted(range(n), key=lambda x: len(triple_partners[x]), reverse=True)
    
    best = 0
    
    # Use iterative deepening / branch and bound
    def dfs(idx, current_size, included_mask):
        nonlocal best
        if current_size > best:
            best = current_size
        
        if idx == n:
            return
        
        # Upper bound: current_size + remaining points
        if current_size + (n - idx) <= best:
            return
        
        i = order[idx]
        
        # Check if adding i is valid
        valid = True
        for (a, b) in triple_partners[i]:
            if (included_mask & (1 << a)) and (included_mask & (1 << b)):
                valid = False
                break
        
        if valid:
            dfs(idx + 1, current_size + 1, included_mask | (1 << i))
        
        dfs(idx + 1, current_size, included_mask)
    
    dfs(0, 0, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
        
    def generate_initial():
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 1000:
            p = random_point()
            if p not in points:
                ok = True
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        if _are_collinear(points[i], points[j], p):
                            for k in range(j+1, len(points)):
                                if _are_collinear(points[i], points[j], points[k]):
                                    ok = False
                                    break
                            if not ok: break
                    if not ok: break
                if ok:
                    points.append(p)
            attempts += 1
        return points

    points = generate_initial()
    if len(points) < 20:
        points = [[x, x*x] for x in range(-10, 10)]
        
    best_score = _max_no_three_collinear(points)
    best_points = [p[:] for p in points]
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        other_indices = [i for i in range(20) if i != idx]
        random.shuffle(other_indices)
        
        moved = False
        for a in range(len(other_indices)):
            i1 = other_indices[a]
            for b in range(a+1, len(other_indices)):
                i2 = other_indices[b]
                p1 = points[i1]
                p2 = points[i2]
                
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = p2[0]*p1[1] - p1[0]*p2[1]
                
                candidates = []
                if A == 0:
                    if C % B == 0:
                        y = -C // B
                        if -100 <= y <= 100:
                            for x in range(-100, 101):
                                if [x, y] not in points and [x, y] != old_p:
                                    candidates.append([x, y])
                elif B == 0:
                    if C % A == 0:
                        x = -C // A
                        if -100 <= x <= 100:
                            for y in range(-100, 101):
                                if [x, y] not in points and [x, y] != old_p:
                                    candidates.append([x, y])
                else:
                    for x in range(-100, 101):
                        num = -(A * x + C)
                        if num % B == 0:
                            y = num // B
                            if -100 <= y <= 100:
                                p_new = [x, y]
                                if p_new not in points and p_new != old_p:
                                    candidates.append(p_new)
                
                if not candidates:
                    continue
                    
                p_new = random.choice(candidates)
                
                points[idx] = p_new
                if _check_no_four_collinear(points):
                    new_score = _max_no_three_collinear(points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = [p[:] for p in points]
                        moved = True
                        break
                    else:
                        points[idx] = old_p
                else:
                    points[idx] = old_p
                    
        if moved:
            continue
            
        for _ in range(10):
            idx = random.randint(0, 19)
            old_p = points[idx]
            p_new = random_point()
            if p_new in points or p_new == old_p:
                continue
            points[idx] = p_new
            if _check_no_four_collinear(points):
                new_score = _max_no_three_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    break
            else:
                points[idx] = old_p

    return best_points
