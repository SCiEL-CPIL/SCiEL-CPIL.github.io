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
    and greedy coloring upper bound.
    """
    n = len(points)
    # Precompute for each vertex v: bitmask of vertices that, together with v,
    # form a collinear triple with another chosen vertex.
    # Actually: for each v, conflict_mask[v] is not directly useful.
    # Better: for each pair (u,w) that are collinear with v, adding v forbids having both u and w.
    
    # Precompute: for each vertex v, the list of (u, w) pairs such that v,u,w collinear and u<w
    # Represent as: for each v, a list of bitmasks where each mask represents a forbidden pair
    # When v is added, we cannot have both bits set in chosen for any forbidden pair.
    
    # Alternative efficient approach:
    # For each vertex v, precompute pair_masks[v] = list of (u_bit | w_bit) for each collinear triple containing v
    # When checking if v can be added: for each mask in pair_masks[v], if (chosen & mask) == mask, conflict.
    
    pair_masks = [[] for _ in range(n)]
    triple_count = [0] * n
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triple_count[i] += 1
                    triple_count[j] += 1
                    triple_count[k] += 1
                    # For vertex i: pair (j,k)
                    pair_masks[i].append((1<<j) | (1<<k))
                    # For vertex j: pair (i,k)
                    pair_masks[j].append((1<<i) | (1<<k))
                    # For vertex k: pair (i,j)
                    pair_masks[k].append((1<<i) | (1<<j))
    
    # Order vertices: hardest first (most triples)
    order = sorted(range(n), key=lambda v: -triple_count[v])
    
    # Remap: new index in order
    old_to_new = [0]*n
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    
    # Rebuild pair_masks in new order
    new_pair_masks = [[] for _ in range(n)]
    for old_v in range(n):
        new_v = old_to_new[old_v]
        for mask in pair_masks[old_v]:
            # Convert mask to new indices
            new_mask = 0
            for u in range(n):
                if (mask >> u) & 1:
                    new_mask |= (1 << old_to_new[u])
            new_pair_masks[new_v].append(new_mask)
    
    best = [0]
    
    def greedy_color_bound(pos, chosen_mask):
        """
        Upper bound on additional vertices we can add from pos onwards.
        Use greedy coloring on the remaining conflict graph.
        A vertex v at position p conflicts with another vertex u at position q>p 
        if adding both would create a collinear triple.
        
        For simplicity, use a looser bound: count remaining vertices minus those
        that directly conflict with chosen vertices.
        """
        # Simple bound: remaining count
        remaining = n - pos
        # Tighter: for each remaining vertex, check if it can be added given current chosen
        # This is expensive, so use a quick estimate
        # Greedy: try to pack as many as possible greedily
        count = 0
        for p in range(pos, n):
            if time.time() > deadline:
                return count  # bail
            v = p
            # Check if v conflicts with any pair in chosen
            conflict = False
            for mask in new_pair_masks[v]:
                if (chosen_mask & mask) == mask:
                    conflict = True
                    break
            if not conflict:
                count += 1
                # Don't actually add it, just count optimistically
        return count
    
    def backtrack(pos, chosen_mask, current_size):
        if time.time() > deadline:
            return
        if pos == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current + remaining
        if current_size + (n - pos) <= best[0]:
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
    Strategy: start with a set of lines, place 3 points on each line, sharing points
    between lines to create a dense structure.
    """
    rng = random.Random(seed)
    
    # Define a set of lines (as (point1, point2) defining the line)
    # We want many lines of exactly 3 points from our 20-point set.
    # Use a combinatorial design approach.
    
    # Approach: use a near-pencil structure with perturbations
    # Base: 20 points roughly on a circle, then adjust to create triples
    
    # Better: use explicit line construction
    # Pick ~10 lines, each with 3 points, with overlaps
    
    points = []
    used = set()
    
    # Define lines by their 3 points. We'll build incrementally.
    # Start with a 5x4 grid-like structure but with specific collinear triples
    
    # Strategy: use affine plane of order 4 (16 points, 20 lines of 4) 
    # but we need 20 points and lines of 3.
    
    # Simpler: construct from scratch
    # Place points to create specific collinear triples
    
    # Use a set of "anchor" lines
    lines = []
    
    # Line 1: horizontal at y=0: (-30,0), (0,0), (30,0)
    # Line 2: horizontal at y=20: (-30,20), (0,20), (30,20)
    # Line 3: horizontal at y=40: (-30,40), (0,40), (30,40)
    # Line 4: horizontal at y=-20: (-30,-20), (0,-20), (30,-20)
    # Line 5: horizontal at y=-40: (-30,-40), (0,-40), (30,-40)
    # That's 15 points. Add 5 more with vertical/diagonal lines.
    
    # Let's try a different approach: use a known configuration
    # 20 points with many 3-collinear triples
    
    # Grid 4x5 with perturbations to break 4-collinear
    base = []
    for row in range(4):
        for col in range(5):
            x = -40 + col * 20
            y = -30 + row * 20
            base.append((x, y))
    
    # Perturb to break 4-collinear while keeping 3-collinear where possible
    points = []
    for (x, y) in base:
        px = x + rng.randint(-3, 3)
        py = y + rng.randint(-3, 3)
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
