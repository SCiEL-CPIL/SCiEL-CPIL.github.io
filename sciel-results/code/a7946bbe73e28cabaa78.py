import numpy as np
import random
import time

def construct():
    # 1. Solver for Max No-3-Collinear Subset
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinear triples
        # points is list of [x, y]
        # triple (i, j, k) is collinear if cross product is 0
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Build adjacency/incidence for faster removal?
        # For N=20, just filtering is fine.
        
        best = 0
        
        def solve(vertices, current_size):
            nonlocal best
            if not vertices:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning: even if we add all remaining vertices, can we beat best?
            if current_size + len(vertices) <= best:
                return
                
            # Pick a triple to branch on.
            # Heuristic: pick a triple with vertices in 'vertices'
            # To speed up, find any triple where all 3 vertices are in 'vertices'
            # If none, we can add all remaining vertices.
            v_set = set(vertices)
            branch_triple = None
            for t in triples:
                if t[0] in v_set and t[1] in v_set and t[2] in v_set:
                    branch_triple = t
                    break
                    
            if not branch_triple:
                # No collinear triples remain in vertices
                new_size = current_size + len(vertices)
                if new_size > best:
                    best = new_size
                return
            
            u, v, w = branch_triple
            
            # Branch 1: Exclude u
            # Branch 2: Exclude v
            # Branch 3: Exclude w
            
            # Optimization: If we exclude u, we remove u from vertices.
            # We don't need to remove v or w, but the triple (u,v,w) is no longer a constraint 
            # because it's not fully contained in the new vertex set.
            # However, other triples involving u are also broken.
            
            verts_list = list(vertices)
            
            # Branch 1: Exclude u
            new_verts_1 = [x for x in verts_list if x != u]
            solve(new_verts_1, current_size)
            
            # Branch 2: Exclude v
            new_verts_2 = [x for x in verts_list if x != v]
            solve(new_verts_2, current_size)
            
            # Branch 3: Exclude w
            new_verts_3 = [x for x in verts_list if x != w]
            solve(new_verts_3, current_size)
            
        solve(list(range(n)), 0)
        return best

    # 2. Validity Check
    def is_valid(points):
        if len(points) != 20:
            return False
        # Check bounds
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        # Check distinct
        if len(set(map(tuple, points))) != 20:
            return False
        # Check no 4 collinear
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) != 0:
                        continue # Not collinear with i,j,k? Wait.
                    # If i,j,k are collinear, check if there is a 4th point l on this line
                    # This is O(N^4) if done naively. 
                    # Better: For each pair (i,j), count points on the line.
                    pass
        
        # Efficient 4-collinear check:
        # For each pair (i, j), calculate the line. Count how many points lie on it.
        # If count >= 4, invalid.
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Reduce direction vector to primitive form to handle scaling?
                # Actually, just check cross product for all other points.
                count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    # 3. Local Search
    random.seed(42)
    
    # Initial configuration: Grid 4x5 scaled
    # We need to break 4-collinearities.
    # Start with a grid and perturb.
    base_x = [0, 10, 20, 30]
    base_y = [0, 10, 20, 30, 40]
    points = []
    for x in base_x:
        for y in base_y:
            points.append([x, y])
            
    # Perturb to break 4-collinearities
    # We will run a loop to fix invalidities
    valid = False
    attempts = 0
    while not valid and attempts < 1000:
        if is_valid(points):
            valid = True
            break
        
        # Find a violation
        # Just pick a random point and move it
        idx = random.randint(0, 19)
        # Move to a random nearby spot
        # Ensure we stay in bounds
        new_x = points[idx][0] + random.randint(-5, 5)
        new_y = points[idx][1] + random.randint(-5, 5)
        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
            # Check if it collides with existing points
            collides = False
            for i, p in enumerate(points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if not collides:
                points[idx] = [new_x, new_y]
        attempts += 1
        
    if not valid:
        # Fallback: Return seed (score 20) if we can't find valid config quickly
        # But grid perturb usually works fast
        return [[x, x * x] for x in range(-10, 10)]

    # Local Search to minimize score
    current_score = max_no_3_collinear(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Simulated Annealing / Hill Climbing
    # Since N=20, score calculation is fast.
    # We can do many iterations.
    time_limit = time.time() + 110 # Leave 10s for safety
    
    while time.time() < time_limit:
        # Pick random point to move
        idx = random.randint(0, 19)
        # Generate random neighbor
        # Move to a random point in a small box around current
        # Or just a random point in the whole grid? 
        # Random whole grid might jump too much.
        # Let's try random nearby first, then random global if stuck.
        step = random.randint(1, 10)
        new_x = points[idx][0] + random.randint(-step, step)
        new_y = points[idx][1] + random.randint(-step, step)
        
        if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
            continue
            
        # Check collision
        collides = False
        for i, p in enumerate(points):
            if i != idx and p[0] == new_x and p[1] == new_y:
                collides = True
                break
        if collides:
            continue
            
        old_p = points[idx][:]
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_score = max_no_3_collinear(points)
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                # Restart timer? No, just keep going
            elif new_score <= best_score:
                # Accept equal or better? 
                # If we accept equal, we might wander. 
                # If we only accept better, we might get stuck in local minima.
                # Let's accept equal with probability 0.1
                if random.random() < 0.1:
                    pass # Keep
                else:
                    points[idx] = old_p
            else:
                # Worse, reject
                points[idx] = old_p
        else:
            points[idx] = old_p

    return best_points

# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    
    # 1. Solver for Max No-3-Collinear Subset
    def max_no_3_collinear(points):
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        best = 0
        
        def solve(vertices, current_size):
            nonlocal best
            if not vertices:
                if current_size > best:
                    best = current_size
                return
            
            if current_size + len(vertices) <= best:
                return
                
            v_set = set(vertices)
            branch_triple = None
            for t in triples:
                if t[0] in v_set and t[1] in v_set and t[2] in v_set:
                    branch_triple = t
                    break
                    
            if not branch_triple:
                new_size = current_size + len(vertices)
                if new_size > best:
                    best = new_size
                return
            
            u, v, w = branch_triple
            verts_list = list(vertices)
            
            # Branch 1: Exclude u
            solve([x for x in verts_list if x != u], current_size)
            # Branch 2: Exclude v
            solve([x for x in verts_list if x != v], current_size)
            # Branch 3: Exclude w
            solve([x for x in verts_list if x != w], current_size)
            
        solve(list(range(n)), 0)
        return best

    # 2. Validity Check
    def is_valid(points):
        if len(points) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if len(set(map(tuple, points))) != 20:
            return False
        
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    # 3. Local Search
    random.seed(42)
    
    base_x = [0, 10, 20, 30]
    base_y = [0, 10, 20, 30, 40]
    points = []
    for x in base_x:
        for y in base_y:
            points.append([x, y])
            
    valid = False
    attempts = 0
    while not valid and attempts < 1000:
        if is_valid(points):
            valid = True
            break
        idx = random.randint(0, 19)
        new_x = points[idx][0] + random.randint(-5, 5)
        new_y = points[idx][1] + random.randint(-5, 5)
        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
            collides = False
            for i, p in enumerate(points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if not collides:
                points[idx] = [new_x, new_y]
        attempts += 1
        
    if not valid:
        return [[x, x * x] for x in range(-10, 10)]

    current_score = max_no_3_collinear(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    time_limit = time.time() + 110
    
    while time.time() < time_limit:
        idx = random.randint(0, 19)
        step = random.randint(1, 10)
        new_x = points[idx][0] + random.randint(-step, step)
        new_y = points[idx][1] + random.randint(-step, step)
        
        if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
            continue
            
        collides = False
        for i, p in enumerate(points):
            if i != idx and p[0] == new_x and p[1] == new_y:
                collides = True
                break
        if collides:
            continue
            
        old_p = points[idx][:]
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_score = max_no_3_collinear(points)
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            elif new_score == best_score:
                if random.random() < 0.1:
                    pass
                else:
                    points[idx] = old_p
            else:
                points[idx] = old_p
        else:
            points[idx] = old_p

    return best_points
# EVOLVE-BLOCK-END
