import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False

    def get_triples(points):
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def exact_independence_number(points):
        n = len(points)
        triples = get_triples(points)
        
        # Precompute for each point, the pairs that form a triple with it
        # pairs_with[i] is a list of (j, k) such that {i, j, k} is a triple
        pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_with[i].append((j, k))
            pairs_with[j].append((i, k))
            pairs_with[k].append((i, j))

        best = 0
        
        # Use bitmask for current set
        # 20 points fit in an int
        def bb(idx, current_mask, current_sum):
            nonlocal best
            if current_sum + (n - idx) <= best:
                return
            
            if idx == n:
                if current_sum > best:
                    best = current_sum
                return
            
            # Try including point idx
            # Check if adding idx creates a triple
            valid = True
            for j, k in pairs_with[idx]:
                # Check if both j and k are in current_mask
                if (current_mask & (1 << j)) and (current_mask & (1 << k)):
                    valid = False
                    break
            
            if valid:
                bb(idx + 1, current_mask | (1 << idx), current_sum + 1)
            
            # Try excluding point idx
            bb(idx + 1, current_mask, current_sum)

        bb(0, 0, 0)
        return best

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_config():
        while True:
            points = [random_point() for _ in range(20)]
            if len(set(map(tuple, points))) < 20:
                continue
            if not has_4_collinear(points):
                return points

    best_points = None
    best_score = 21  # Max possible is 20
    
    # Initial random search
    for _ in range(20):
        if time.time() - start_time > timeout:
            break
        pts = generate_valid_config()
        score = exact_independence_number(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = generate_valid_config()
        best_score = exact_independence_number(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Hill Climbing
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
        
        # Perturb if not first
        if restart > 0:
            # Pick a random point and move it
            idx = random.randint(0, 19)
            current_points[idx] = random_point()
            attempts = 0
            while has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20:
                current_points[idx] = random_point()
                attempts += 1
                if attempts > 100:
                    break
            
            if has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20:
                continue

            current_score = exact_independence_number(current_points)
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Try moving each point
            for i in range(20):
                if time.time() - start_time > timeout:
                    break
                orig_point = current_points[i]
                # Try a few random moves
                for _ in range(5):
                    new_point = random_point()
                    if new_point == orig_point:
                        continue
                    current_points[i] = new_point
                    if len(set(map(tuple, current_points))) < 20:
                        current_points[i] = orig_point
                        continue
                    if has_4_collinear(current_points):
                        current_points[i] = orig_point
                        continue
                    
                    new_score = exact_independence_number(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        break
                    else:
                        current_points[i] = orig_point
                        # If we didn't improve, we might still want to try another move for this point?
                        # Or break to save time?
                        # Let's try a few more.
                if improved:
                    break

    return best_points
