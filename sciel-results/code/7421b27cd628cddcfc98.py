def construct():
    import random
    import numpy as np
    
    N = 16
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = N * N
    point_indices = {p: i for i, p in enumerate(all_points)}
    
    def is_isosceles(a, b, c):
        d1 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        d2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        d3 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return d1 == d2 or d1 == d3 or d2 == d3

    def is_valid_no_isosceles(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if is_isosceles(points[i], points[j], points[k]):
                        return False
        return True

    def is_maximal(points, all_pts):
        point_set = set(points)
        for p in all_pts:
            if p in point_set:
                continue
            # Check if p can be added
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if is_isosceles(p, points[i], points[j]):
                        break
                else:
                    continue
                break
            else:
                return False  # p can be added, so not maximal
        return True

    def generate_initial_set(trial):
        random.seed(trial)
        perm = all_points[:]
        random.shuffle(perm)
        chosen = []
        for p in perm:
            can_add = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_isosceles(p, chosen[i], chosen[j]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen.append(p)
        return chosen

    def try_improve(S):
        S_set = set(S)
        S_list = list(S)
        improved = False
        # Try swapping 2 points for 1
        # This is expensive, so we limit trials
        for _ in range(10):
            if len(S_list) < 2:
                break
            p1_idx = random.randint(0, len(S_list)-1)
            p2_idx = random.randint(0, len(S_list)-1)
            if p1_idx == p2_idx:
                continue
            p1 = S_list[p1_idx]
            p2 = S_list[p2_idx]
            
            # Try to find a q outside S that allows removing p1, p2
            # q must be valid with S \ {p1, p2}
            # and must block p1 and p2 (or other points)
            
            candidates = [p for p in all_points if p not in S_set]
            random.shuffle(candidates)
            
            for q in candidates[:20]: # Limit search
                S_new = S_set - {p1, p2} | {q}
                S_new_list = list(S_new)
                
                # Check validity
                if not is_valid_no_isosceles(S_new_list):
                    continue
                
                # Check maximality
                if is_maximal(S_new_list, all_points):
                    return S_new_list, True
        
        return S_list, False

    best_set = None
    best_size = n_points + 1
    
    # Generate several initial sets
    for trial in range(50):
        S = generate_initial_set(trial)
        if not is_maximal(S, all_points):
            continue
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Try to improve
        S_improved, improved = try_improve(S)
        if improved and len(S_improved) < best_size:
            best_size = len(S_improved)
            best_set = S_improved[:]

    return best_set if best_set else all_points
