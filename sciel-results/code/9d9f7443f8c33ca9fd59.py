import itertools
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for final scoring and return

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_triples(points):
        """Counts the number of collinear triples in the point set."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def has_four_collinear(points):
        """Checks if any 4 points are collinear. Returns True if invalid."""
        n = len(points)
        # For each pair, count how many points lie on the line defined by them
        # If any line has >= 4 points, it's invalid.
        # To avoid O(N^4), we can use a set of normalized lines or just check pairs.
        # N=20, so N^2=400 pairs. For each pair, checking N-2 points is 18. 400*18 = 7200 ops. Fast.
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calc_score(points):
        """
        Exact branch and bound for the largest subset with no three collinear.
        """
        n = len(points)
        triples = set()
        for i, j, k in itertools.combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples.add((i, j, k))
        
        # Precompute pairs for each vertex
        pairs_for_vertex = [set() for _ in range(n)]
        for i, j, k in triples:
            pairs_for_vertex[i].add((j, k))
            pairs_for_vertex[j].add((i, k))
            pairs_for_vertex[k].add((i, j))
            
        best_size = 0
        chosen = [False] * n
        
        def backtrack(idx, current_size):
            nonlocal best_size
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pruning
            if current_size + (n - idx) <= best_size:
                return
                
            # Option 1: Exclude points[idx]
            backtrack(idx + 1, current_size)
            
            # Option 2: Include points[idx]
            v = idx
            valid_inclusion = True
            # Check if any pair in chosen forms a triple with v
            # pairs_for_vertex[v] contains pairs (u, w) such that {v, u, w} is a triple.
            # We need to check if u and w are BOTH in chosen.
            for u, w in pairs_for_vertex[v]:
                if chosen[u] and chosen[w]:
                    valid_inclusion = False
                    break
                    
            if valid_inclusion:
                chosen[v] = True
                backtrack(idx + 1, current_size + 1)
                chosen[v] = False
        
        backtrack(0, 0)
        return best_size

    def generate_initial_line_packing():
        """
        Generates a starting configuration with many collinear triples
        by greedily adding points on lines defined by existing points.
        """
        points = []
        # Start with 3 random points to form the first line
        while len(points) < 3:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in points:
                points.append(p)
        
        while len(points) < 20:
            # Pick a random existing pair
            i, j = random.sample(range(len(points)), 2)
            p1, p2 = points[i], points[j]
            
            # Find integer points on the line p1-p2
            # Vector (dx, dy)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0: continue # Should not happen for distinct points
            
            step_x = dx // g
            step_y = dy // g
            
            candidates = []
            # Extend in both directions
            for k in range(1, 100):
                nx1 = p1[0] + k * step_x
                ny1 = p1[1] + k * step_y
                if -100 <= nx1 <= 100 and -100 <= ny1 <= 100 and (nx1, ny1) not in points:
                    candidates.append((nx1, ny1))
                
                nx2 = p1[0] - k * step_x
                ny2 = p1[1] - k * step_y
                if -100 <= nx2 <= 100 and -100 <= ny2 <= 100 and (nx2, ny2) not in points:
                    candidates.append((nx2, ny2))
            
            if candidates:
                # Pick a random candidate
                p_new = random.choice(candidates)
                points.append(p_new)
            else:
                # If no candidates on this line, pick a random point
                p_new = (random.randint(-100, 100), random.randint(-100, 100))
                if p_new not in points:
                    points.append(p_new)
                else:
                    # Fallback: keep trying random points
                    while True:
                        p_new = (random.randint(-100, 100), random.randint(-100, 100))
                        if p_new not in points:
                            points.append(p_new)
                            break
        return points

    def local_search(points, time_limit):
        """
        Simulated Annealing to maximize collinear triples while maintaining validity.
        """
        current_points = points[:]
        current_triples = count_triples(current_points)
        
        # Check validity of initial state, if invalid, fix it or restart
        if has_four_collinear(current_points):
             # Try to fix by randomizing points involved in 4-collinear sets
             # Or just randomize the whole set if it's bad
             # For now, if invalid, we will rely on SA to fix it, but SA rejects invalid states.
             # So we must start valid.
             # Let's force validity in the initializer or here.
             pass

        best_points = current_points[:]
        best_triples = current_triples
        best_score = 20 # Placeholder, will compute later
        
        T = 100.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < time_limit and T > T_min:
            # Generate a move
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            # Smart move: try to create a triple
            # Pick two other random points
            others = [i for i in range(20) if i != idx]
            if len(others) >= 2:
                j, k = random.sample(others, 2)
                p_j, p_k = current_points[j], current_points[k]
                
                # Find points on line p_j-p_k
                dx = p_k[0] - p_j[0]
                dy = p_k[1] - p_j[1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    step_x = dx // g
                    step_y = dy // g
                    # Try a few points on this line
                    found_candidate = False
                    for offset in random.sample(range(1, 50), 5):
                        # Check both directions from p_j
                        for sign in [1, -1]:
                            nx = p_j[0] + sign * offset * step_x
                            ny = p_j[1] + sign * offset * step_y
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                new_p = (nx, ny)
                                if new_p not in current_points:
                                    # Try this move
                                    current_points[idx] = new_p
                                    if not has_four_collinear(current_points):
                                        new_triples = count_triples(current_points)
                                        delta = new_triples - current_triples
                                        if delta > 0 or random.random() < math.exp(delta / T):
                                            current_triples = new_triples
                                            found_candidate = True
                                            break
                                    else:
                                        current_points[idx] = old_p
                        if found_candidate:
                            break
                    
                    if not found_candidate:
                         # Revert if smart move failed or wasn't accepted
                         if current_points[idx] != old_p:
                             current_points[idx] = old_p
                         else:
                             # Try a random move
                             new_p = (random.randint(-100, 100), random.randint(-100, 100))
                             if new_p not in current_points:
                                 current_points[idx] = new_p
                                 if not has_four_collinear(current_points):
                                     new_triples = count_triples(current_points)
                                     delta = new_triples - current_triples
                                     if delta > 0 or random.random() < math.exp(delta / T):
                                         current_triples = new_triples
                                     else:
                                         current_points[idx] = old_p
                else:
                    # p_j == p_k? No, distinct points.
                    pass
            else:
                # Random move
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p not in current_points:
                    current_points[idx] = new_p
                    if not has_four_collinear(current_points):
                        new_triples = count_triples(current_points)
                        delta = new_triples - current_triples
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_triples = new_triples
                        else:
                            current_points[idx] = old_p
                    else:
                        current_points[idx] = old_p
            
            # Track best
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = current_points[:]
            
            T *= cooling
        
        return best_points, best_triples

    # Main Loop
    best_points = None
    best_exact_score = 21
    
    # We run multiple restarts
    num_restarts = 500 # Will be limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Generate initial config
        init_points = generate_initial_line_packing()
        
        # Ensure validity
        if has_four_collinear(init_points):
            # Simple repair: randomize points until valid
            valid = False
            for _ in range(100):
                if not has_four_collinear(init_points):
                    valid = True
                    break
                idx = random.randint(0, 19)
                init_points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
            if not valid:
                continue
        
        # 2. Local Search
        # Allocate time: 0.5s per restart?
        # If we have 110s, 200 restarts of 0.5s is good.
        ls_time = 0.5
        ls_points, _ = local_search(init_points, time.time() + ls_time)
        
        # 3. Evaluate Exact Score
        # Only evaluate if the triple count is promising?
        # Or just evaluate all? B&B is fast for N=20.
        score = calc_score(ls_points)
        
        if score < best_exact_score:
            best_exact_score = score
            best_points = ls_points
            # If we find a very good score, we can try to polish it further
            if best_exact_score <= 10:
                # Extend search on this best
                # Run more LS on this specific best
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    ls_points, _ = local_search(best_points, time.time() + 0.2)
                    score = calc_score(ls_points)
                    if score < best_exact_score:
                        best_exact_score = score
                        best_points = ls_points

    if best_points is None:
        # Fallback
        best_points = [[x, x*x] for x in range(-10, 10)]
        # Ensure validity for fallback (parabola has no 3 collinear, so no 4 collinear)
        # Score is 20.

    return [[p[0], p[1]] for p in best_points]
