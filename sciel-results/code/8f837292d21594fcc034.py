import time
import random
import sys
from math import gcd
from collections import defaultdict

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            if A == 0 and B == 0:
                continue
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                if lines[key] >= 4:
                    return False
    return True


def _max_independent_set(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n == 0:
        return 0

    # Precompute: for each pair (i,j), which other points k are collinear with them
    collinear_with = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0

    def backtrack(current_mask, candidates_mask):
        nonlocal best
        cand_count = candidates_mask.bit_count()
        cur_count = current_mask.bit_count()
        if cur_count + cand_count <= best:
            return
        if cand_count == 0:
            if cur_count > best:
                best = cur_count
            return

        # Pick point with highest conflict degree among candidates for better pruning
        # Find the candidate with the most conflicts
        best_idx = -1
        best_conflicts = -1
        cm = candidates_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            # Count conflicts: how many other candidates are collinear with r_idx and some selected point
            # Actually, just use degree in the conflict graph among candidates
            conflict_count = 0
            cm2 = candidates_mask & ~(1 << r_idx)
            while cm2:
                s_idx = (cm2 & -cm2).bit_length() - 1
                # Check if r_idx and s_idx are collinear with any selected point
                # For simplicity, just count direct conflicts
                conflict_count += 1
                cm2 &= cm2 - 1
            # Use a simpler heuristic: pick the first one
            best_idx = r_idx
            break
        
        p_idx = best_idx
        rest_mask = candidates_mask & ~(1 << p_idx)

        # Branch 1: Include p_idx
        blocked_mask = 0
        cm = current_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            blocked_mask |= collinear_with[p_idx][r_idx]
            cm &= cm - 1

        new_candidates = rest_mask & ~blocked_mask
        backtrack(current_mask | (1 << p_idx), new_candidates)

        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask)

    backtrack(0, (1 << n) - 1)
    return best


def _count_triples(points):
    """Count total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
    return count


def _generate_candidate_pool(size=500):
    """
    Generate a pool of candidate points that are likely to form collinear triples.
    Strategy: Generate points along many different lines, ensuring diversity.
    """
    random.seed(42)
    pool = set()
    
    # Generate points along many random lines
    for _ in range(200):
        # Random line through two random points
        x1 = random.randint(-50, 50)
        y1 = random.randint(-50, 50)
        x2 = random.randint(-50, 50)
        y2 = random.randint(-50, 50)
        
        # Add several points along this line (within bounds)
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(gcd(abs(dx), abs(dy)), 1)
        if g == 0:
            continue
        dx //= g
        dy //= g
        
        # Add points at various positions along the line
        for t in range(-3, 4):
            x = x1 + t * dx
            y = y1 + t * dy
            if -100 <= x <= 100 and -100 <= y <= 100:
                pool.add((x, y))
    
    # Also add some grid-like points for structure
    for i in range(-20, 21, 4):
        for j in range(-20, 21, 4):
            if random.random() < 0.5:
                pool.add((i, j))
    
    # Convert to list
    return list(pool)


def _compute_triple_scores(candidate_pool, all_points_indices):
    """
    For each candidate point, compute how many collinear triples it would form
    with other candidate points.
    """
    n = len(candidate_pool)
    scores = [0] * n
    
    # For each pair, count how many third points are collinear
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = candidate_pool[i]
            x2, y2 = candidate_pool[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Count points collinear with i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = candidate_pool[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    # This triple (i, j, k) contributes to all three points
                    scores[i] += 1
                    scores[j] += 1
                    scores[k] += 1
    
    return scores


def _greedy_select(candidate_pool, target_size=20, seed=0):
    """
    Greedy selection: repeatedly pick the point that maximizes the number of
    new collinear triples formed, while maintaining the no-four-collinear constraint.
    """
    random.seed(seed)
    selected = []
    selected_set = set()
    
    # Precompute all collinear triples in the candidate pool
    n = len(candidate_pool)
    triples = []  # List of (i, j, k) indices
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = candidate_pool[i]
            x2, y2 = candidate_pool[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = candidate_pool[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    triples.append((i, j, k))
    
    # For each triple, track which points it involves
    # Greedy: at each step, pick the point that completes the most new triples
    # with already-selected points
    
    for _ in range(target_size):
        best_point = -1
        best_score = -1
        best_triples_count = 0
        
        for idx in range(n):
            if idx in selected_set:
                continue
            
            # Count how many new triples this point would form with selected points
            new_triples = 0
            for triple in triples:
                if idx in triple:
                    # Check if the other two points are in selected
                    other1, other2 = [t for t in triple if t != idx]
                    if other1 in selected_set and other2 in selected_set:
                        new_triples += 1
            
            if new_triples > best_score or (new_triples == best_score and random.random() < 0.3):
                best_score = new_triples
                best_point = idx
                best_triples_count = new_triples
        
        if best_point == -1:
            break
        
        # Check validity: adding this point doesn't create 4-collinear
        candidate = candidate_pool[best_point]
        test_points = selected + [candidate]
        if _check_no_four_collinear(test_points):
            selected.append(candidate)
            selected_set.add(best_point)
        else:
            # Try next best point
            # For simplicity, skip and try again (this is a simplification)
            pass
    
    return selected


def _local_refinement(points, time_limit):
    """
    Local refinement: try to improve the configuration by swapping points
    to increase triple count while maintaining validity.
    """
    current_pts = [p[:] for p in points]
    current_triples = _count_triples(current_pts)
    best_pts = [p[:] for p in current_pts]
    best_triples = current_triples
    
    start = time.time()
    n = 20
    
    while time.time() - start < time_limit:
        # Pick a random point to try to improve
        idx = random.randint(0, n - 1)
        ox, oy = current_pts[idx]
        
        # Try moving to a nearby position
        improved = False
        for _ in range(10):
            radius = random.choice([2, 3, 5, 8])
            nx = ox + random.randint(-radius, radius)
            ny = oy + random.randint(-radius, radius)
            
            if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                continue
            
            new_p = [nx, ny]
            if new_p == current_pts[idx]:
                continue
            
            # Check duplicate
            dup = False
            for i in range(n):
                if i != idx and current_pts[i] == new_p:
                    dup = True
                    break
            if dup:
                continue
            
            # Test the move
            old_p = current_pts[idx]
            current_pts[idx] = new_p
            
            if not _check_no_four_collinear(current_pts):
                current_pts[idx] = old_p
                continue
            
            new_triples = _count_triples(current_pts)
            
            if new_triples > current_triples:
                current_triples = new_triples
                improved = True
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_pts = [p[:] for p in current_pts]
            else:
                current_pts[idx] = old_p
            
            if improved:
                break
        
        if not improved:
            # Try a different point
            continue
    
    return best_pts, best_triples


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_points = None
    best_score = float('inf')
    best_triples = -1
    
    # Generate candidate pool
    candidate_pool = _generate_candidate_pool(500)
    
    # Run multiple greedy selections with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        selected = _greedy_select(candidate_pool, 20, seed)
        
        if len(selected) < 20:
            continue
        
        if not _check_no_four_collinear(selected):
            continue
        
        triples = _count_triples(selected)
        
        if triples > best_triples:
            best_triples = triples
            best_points = selected
            print(f"Greedy seed {seed}: triples={triples}", file=sys.stderr)
    
    # Local refinement on the best configuration
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            refined_pts, refined_triples = _local_refinement(best_points, min(remaining - 10, 60))
            if refined_triples > best_triples:
                best_triples = refined_triples
                best_points = refined_pts
                print(f"Refined: triples={refined_triples}", file=sys.stderr)
    
    # Compute final score
    if best_points is None:
        # Fallback: simple valid configuration
        random.seed(0)
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts and _check_no_four_collinear(pts + [p]):
                pts.append(p)
        best_points = pts
    
    # Verify and return
    assert len(best_points) == 20, f"Expected 20 points, got {len(best_points)}"
    assert _check_no_four_collinear(best_points), "Configuration has 4 collinear points"
    
    return best_points


if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(f"Triples: {_count_triples(pts)}")
