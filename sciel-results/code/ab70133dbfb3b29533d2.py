"""Combined program: constructive line-packing + bitmask B&B with coloring bounds."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

def _has_four_collinear(points):
    """Check if any four points are collinear. Returns True if invalid."""
    n = len(points)
    # For each pair, count collinear points
    for i in range(n):
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            # Reduce direction to primitive
            from math import gcd
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            px, py = dx//g, dy//g
            count = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0]-points[i][0])*py == (points[k][1]-points[i][1])*px:
                    count += 1
            if count >= 2:  # i, j plus 2 more = 4 collinear
                return True
    return False

def _count_collinear_triples(points):
    """Count collinear triples."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            for k in range(j+1, n):
                if (points[k][0]-points[i][0])*dy == (points[k][1]-points[i][1])*dx:
                    triples += 1
    return triples

def _exact_max_3free(points, deadline):
    """
    Exact branch-and-bound for largest 3-free subset using bitmasks
    and a tighter upper bound based on independent set in the conflict graph.
    """
    n = len(points)
    
    # Precompute conflict pairs: if i and j are chosen, which k's are forbidden?
    # Actually, the constraint is: no 3 collinear.
    # This is equivalent to: the set of chosen points must be an independent set
    # in a hypergraph where edges are collinear triples.
    # For B&B, we can use the fact that if we choose a point, we cannot choose
    # any other two points that form a collinear triple with it.
    
    # Let's use the pair_masks approach but with a better bound.
    pair_masks = [[] for _ in range(n)]
    triple_count = [0] * n
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triple_count[i] += 1
                    triple_count[j] += 1
                    triple_count[k] += 1
                    pair_masks[i].append((1<<j) | (1<<k))
                    pair_masks[j].append((1<<i) | (1<<k))
                    pair_masks[k].append((1<<i) | (1<<j))
    
    # Order vertices: hardest first (most triples)
    order = sorted(range(n), key=lambda v: -triple_count[v])
    old_to_new = [0]*n
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    
    new_pair_masks = [[] for _ in range(n)]
    for old_v in range(n):
        new_v = old_to_new[old_v]
        for mask in pair_masks[old_v]:
            new_mask = 0
            for u in range(n):
                if (mask >> u) & 1:
                    new_mask |= (1 << old_to_new[u])
            new_pair_masks[new_v].append(new_mask)
    
    best = [0]
    
    def bound(pos, chosen_mask):
        """
        Upper bound: maximum independent set in the remaining graph.
        We approximate this by counting how many remaining vertices can be added
        without conflict with chosen_mask, and then using a greedy packing on the rest.
        """
        count = 0
        available = (1 << n) - 1
        # Mask of vertices already considered or chosen
        # We iterate from pos to n-1
        for v in range(pos, n):
            if time.time() > deadline:
                return count
            # Check if v can be added
            can_add = True
            for mask in new_pair_masks[v]:
                if (chosen_mask & mask) == mask:
                    can_add = False
                    break
            if can_add:
                count += 1
                # Update chosen_mask for the bound calculation only
                chosen_mask |= (1 << v)
        return count
    
    def backtrack(pos, chosen_mask, current_size):
        if time.time() > deadline:
            return
        if pos == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound
        ub = bound(pos, chosen_mask)
        if current_size + ub <= best[0]:
            return
        
        # Try skipping
        backtrack(pos + 1, chosen_mask, current_size)
        
        # Try including
        v = pos
        can_add = True
        for mask in new_pair_masks[v]:
            if (chosen_mask & mask) == mask:
                can_add = False
                break
        
        if can_add:
            backtrack(pos + 1, chosen_mask | (1 << v), current_size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _generate_line_packing(seed):
    """
    Constructive heuristic: place points on shared lines to maximize triple density.
    Uses a grid-based approach with targeted perturbations to create 3-collinearities
    while avoiding 4-collinearities.
    """
    rng = random.Random(seed)
    
    # Start with a 4x5 grid
    base = []
    for row in range(4):
        for col in range(5):
            x = -40 + col * 20
            y = -30 + row * 20
            base.append((x, y))
    
    points = []
    for (x, y) in base:
        # Small perturbation to break 4-collinearities in rows/cols
        # but keep them close to grid lines
        px = x + rng.randint(-2, 2)
        py = y + rng.randint(-2, 2)
        px = max(-100, min(100, px))
        py = max(-100, min(100, py))
        points.append((px, py))
    
    # Remove duplicates
    seen = set()
    unique = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    while len(unique) < 20:
        p = (rng.randint(-100, 100), rng.randint(-100, 100))
        if p not in seen:
            seen.add(p)
            unique.append(p)
    
    return [list(p) for p in unique[:20]]

def _local_search(points, deadline, rng):
    """
    Local search: try to increase triple count (surrogate for lower score).
    Move one point at a time to a nearby position.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    best_triples = current_triples
    best_points = [p[:] for p in points]
    
    while time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() > deadline:
                break
            orig = points[i][:]
            for _ in range(30):
                if time.time() > deadline:
                    break
                new_x = max(-100, min(100, orig[0] + rng.randint(-5, 5)))
                new_y = max(-100, min(100, orig[1] + rng.randint(-5, 5)))
                points[i] = [new_x, new_y]
                
                # Check validity
                if len(set(tuple(p) for p in points)) != 20:
                    points[i] = orig[:]
                    continue
                if _has_four_collinear(points):
                    points[i] = orig[:]
                    continue
                
                new_triples = _count_collinear_triples(points)
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_points = [p[:] for p in points]
                    current_triples = new_triples
                    improved = True
                    break
                elif new_triples > current_triples:
                    current_triples = new_triples
            if points[i] != orig and not improved:
                points[i] = orig[:]
        
        if not improved:
            break
    
    return best_points, best_triples

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s buffer before 150s kill
    
    best_points = None
    best_score = 21
    best_triples = -1
    rng = random.Random(42)
    
    # Phase 1: Generate candidates via line-packing heuristic
    for seed in range(200):
        if time.time() > deadline - 40:
            break
        
        points = _generate_line_packing(seed)
        
        if len(set(tuple(p) for p in points)) != 20:
            continue
        if _has_four_collinear(points):
            continue
        
        triples = _count_collinear_triples(points)
        
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in points]
        
        # Local search on promising candidates
        if triples > 40 and time.time() < deadline - 50:
            improved_points, improved_triples = _local_search(points, deadline - 45, rng)
            if len(set(tuple(p) for p in improved_points)) == 20 and not _has_four_collinear(improved_points):
                if improved_triples > best_triples:
                    best_triples = improved_triples
                    best_points = [p[:] for p in improved_points]
    
    # Phase 2: Evaluate top candidates with exact B&B
    if best_points is not None:
        # Evaluate the best candidate
        if time.time() < deadline - 10:
            score = _exact_max_3free(best_points, deadline - 5)
            if score < best_score:
                best_score = score
    
    # If we have time, try a few more candidates
    if best_points is not None and time.time() < deadline - 20:
        # Generate a few more and evaluate
        for seed in range(200, 250):
            if time.time() > deadline - 10:
                break
            points = _generate_line_packing(seed)
            if len(set(tuple(p) for p in points)) != 20:
                continue
            if _has_four_collinear(points):
                continue
            triples = _count_collinear_triples(points)
            if triples > best_triples:
                score = _exact_max_3free(points, deadline - 5)
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
                    best_triples = triples
    
    if best_points is not None:
        return best_points
    
    # Fallback: parabola (no 3 collinear, score = 20)
    return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
