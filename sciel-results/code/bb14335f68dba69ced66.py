import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s margin before 150s limit
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def count_max_collinear(points):
        """Find the maximum number of collinear points in the set."""
        n = len(points)
        max_count = 2
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count > max_count:
                    max_count = count
        return max_count
    
    def has_4_collinear(points):
        """Check if any 4 points are collinear."""
        return count_max_collinear(points) >= 4
    
    def count_triples(points):
        """Count number of collinear triples."""
        n = len(points)
        triples = 0
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples += 1
        return triples
    
    def get_triples_list(points):
        """Get list of all collinear triples as (i, j, k)."""
        n = len(points)
        triples = []
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                triples.append((i, j, k))
        return triples
    
    def exact_independence_number(points):
        """
        Exact branch and bound for maximum subset with no 3 collinear.
        This is the score we want to minimize.
        """
        n = len(points)
        triples = get_triples_list(points)
        
        # pairs_with[i]: for each point i, list of (j, k) such that {i, j, k} is a triple
        pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_with[i].append((j, k))
            pairs_with[j].append((i, k))
            pairs_with[k].append((i, j))
        
        # Sort points by number of triples they participate in (descending)
        # This helps pruning
        degree = [len(pairs_with[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degree[i])
        
        # Remap: new_index -> old_index
        remap = [0] * n
        for new_idx, old_idx in enumerate(order):
            remap[new_idx] = old_idx
        
        # Rebuild pairs_with in new ordering
        new_pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            ni = order.index(i)
            nj = order.index(j)
            nk = order.index(k)
            new_pairs_with[ni].append((nj, nk))
            new_pairs_with[nj].append((ni, nk))
            new_pairs_with[nk].append((ni, nj))
        
        best = 0
        
        def bb(idx, current_mask, current_sum):
            nonlocal best
            # Upper bound: current_sum + remaining points
            if current_sum + (n - idx) <= best:
                return
            
            if idx == n:
                if current_sum > best:
                    best = current_sum
                return
            
            # Try including point idx
            valid = True
            for j, k in new_pairs_with[idx]:
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
        """Generate a random valid configuration (no 4 collinear)."""
        while True:
            points = [random_point() for _ in range(20)]
            if len(set(map(tuple, points))) < 20:
                continue
            if not has_4_collinear(points):
                return points
    
    def generate_structured_config():
        """
        Generate a structured configuration that encourages collinear triples.
        Strategy: place points on a few lines, but ensure no 4 are on the same line.
        """
        # Use a grid-like structure with some perturbation
        # Place points in clusters near a few lines
        lines = []
        # Choose a few line directions
        num_lines = random.randint(5, 8)
        for _ in range(num_lines):
            # Random line through origin with small slope
            if random.random() < 0.3:
                # Horizontal
                y = random.randint(-90, 90)
                x1 = random.randint(-90, -60)
                x2 = random.randint(60, 90)
                line_points = [(x, y) for x in range(x1, x2, random.randint(15, 25))]
            elif random.random() < 0.6:
                # Vertical
                x = random.randint(-90, 90)
                y1 = random.randint(-90, -60)
                y2 = random.randint(60, 90)
                line_points = [(x, y) for y in range(y1, y2, random.randint(15, 25))]
            else:
                # Diagonal
                m = random.choice([1, -1, 2, -2, 1/2, -1/2])
                b = random.randint(-50, 50)
                line_points = []
                for x in range(-90, 91, random.randint(15, 25)):
                    y = int(m * x + b)
                    if abs(y) <= 90:
                        line_points.append((x, y))
            lines.append(line_points)
        
        # Collect candidate points
        candidates = []
        for line in lines:
            candidates.extend(line)
        
        # Add some random points
        for _ in range(random.randint(5, 10)):
            candidates.append(tuple(random_point()))
        
        # Shuffle and pick 20 distinct points
        random.shuffle(candidates)
        points = []
        seen = set()
        for c in candidates:
            if c not in seen:
                points.append(list(c))
                seen.add(c)
            if len(points) == 20:
                break
        
        # If not enough, fill with random
        while len(points) < 20:
            p = random_point()
            if tuple(p) not in seen:
                points.append(p)
                seen.add(tuple(p))
        
        # Check validity
        if has_4_collinear(points):
            # Perturb to fix
            for i in range(20):
                if count_max_collinear(points) < 4:
                    break
                # Find which line has 4
                for j in range(20):
                    if j == i:
                        continue
                    count = 2
                    for k in range(20):
                        if k != i and k != j:
                            if are_collinear(points[i], points[j], points[k]):
                                count += 1
                    if count >= 4:
                        # Move point i
                        points[i] = random_point()
                        break
            if has_4_collinear(points):
                return None
            if len(set(map(tuple, points))) < 20:
                return None
        return points
    
    best_points = None
    best_score = 21
    best_triple_count = 0
    
    def evaluate_and_update(points):
        nonlocal best_points, best_score, best_triple_count
        triples = count_triples(points)
        score = exact_independence_number(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            best_triple_count = triples
        elif score == best_score and triples > best_triple_count:
            # Same score but more triples - might be better for further optimization
            # Keep the one with more triples as current working point
            pass
        return score, triples
    
    # Phase 1: Generate diverse starting points
    initial_configs = []
    
    # Random configs
    for _ in range(15):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_valid_config()
        initial_configs.append(pts)
    
    # Structured configs
    for _ in range(20):
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_structured_config()
        if pts is not None:
            initial_configs.append(pts)
    
    # Evaluate all initial configs
    for pts in initial_configs:
        if time.time() - start_time > timeout - 20:
            break
        score, triples = evaluate_and_update(pts)
    
    if best_points is None:
        best_points = generate_valid_config()
        best_score, best_triple_count = evaluate_and_update(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    current_triples = best_triple_count
    
    # Phase 2: Local search with hill climbing
    # Objective: minimize independence number, tie-break by maximizing triples
    for restart in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        # If not first restart, perturb
        if restart > 0:
            # Pick 1-3 random points and move them
            num_moves = random.randint(1, 3)
            for _ in range(num_moves):
                idx = random.randint(0, 19)
                current_points[idx] = random_point()
            
            # Validate
            attempts = 0
            while (has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20) and attempts < 50:
                # Fix by moving a random point
                idx = random.randint(0, 19)
                current_points[idx] = random_point()
                attempts += 1
            
            if has_4_collinear(current_points) or len(set(map(tuple, current_points))) < 20:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                current_triples = best_triple_count
                continue
            
            current_score = exact_independence_number(current_points)
            current_triples = count_triples(current_points)
        
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            
            # Try moving each point
            for i in range(20):
                if time.time() - start_time > timeout - 5:
                    break
                
                orig_point = current_points[i]
                
                # Try several moves for this point
                for _ in range(8):
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
                    
                    new_triples = count_triples(current_points)
                    
                    # We want to minimize independence number, but that's expensive.
                    # Use triples as a proxy: more triples -> likely smaller independence number
                    if new_triples > current_triples:
                        # Likely improvement - verify with exact
                        new_score = exact_independence_number(current_points)
                        if new_score < current_score:
                            current_score = new_score
                            current_triples = new_triples
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                                best_triple_count = new_triples
                            break
                        elif new_score == current_score:
                            # Same score, more triples - accept
                            current_triples = new_triples
                            improved = True
                            break
                        else:
                            current_points[i] = orig_point
                    elif new_triples == current_triples:
                        # Sideways move - accept with small probability
                        if random.random() < 0.1:
                            improved = True
                            break
                        else:
                            current_points[i] = orig_point
                    else:
                        current_points[i] = orig_point
                
                if improved:
                    break
        
        # After local search, do exact evaluation of current point
        current_score = exact_independence_number(current_points)
        current_triples = count_triples(current_points)
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            best_triple_count = current_triples
        elif current_score == best_score and current_triples > best_triple_count:
            best_triple_count = current_triples
    
    # Final: return the best point found
    # Do one final exact check on best_points
    final_score = exact_independence_number(best_points)
    if final_score < best_score:
        best_score = final_score
    
    return best_points
