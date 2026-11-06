"""Rewritten program: line-centric genetic algorithm for 16-point 4-collinear maximization."""
import time
import random
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START
import sys
import math
from itertools import combinations as _comb

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Canonical key for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    # canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # normalize anchor to the "lowest" point in direction order
    # line: (x,y) = (x1,y1) + t*(dx,dy); choose anchor with minimal t among the two
    # For canonical key, use (dx,dy, c) where c = dy*x1 - dx*y1 (line equation: dy*x - dx*y = c)
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _line_key3(p1, p2, p3):
    """Line key given 3 collinear points (redundant check)."""
    return _line_key(p1, p2)

def _score_points(pts):
    """Count lines containing exactly 4 of the points. Return (count, lines_dict)."""
    if len(pts) != 16:
        return 0, {}
    # map each point to index
    pset = set(pts)
    # group points by line
    lines = defaultdict(set)
    for i in range(16):
        for j in range(i + 1, 16):
            k = _line_key(pts[i], pts[j])
            if k is None:
                continue
            lines[k].add(i)
    exact4 = 0
    valid_lines = {}
    for k, idxs in lines.items():
        if len(idxs) == 4:
            exact4 += 1
            valid_lines[k] = idxs
    # check no five collinear
    for k, idxs in lines.items():
        if len(idxs) >= 5:
            return 0, {}
    return exact4, valid_lines

def _extract_lines(pts):
    """Return list of sets of point-tuples forming 4-collinear lines, plus check validity."""
    pset = set(pts)
    lines_map = defaultdict(set)
    for i in range(16):
        for j in range(i + 1, 16):
            k = _line_key(pts[i], pts[j])
            if k is None:
                continue
            lines_map[k].add(i)
    out = []
    for k, idxs in lines_map.items():
        if len(idxs) >= 5:
            return None  # invalid
        if len(idxs) == 4:
            out.append([pts[i] for i in idxs])
    return out

def _line_points_set(line_pts):
    return frozenset(line_pts)

def _try_add_point_to_lines(lines, pt, existing_pts):
    """Given a set of 4-point lines (as lists of point tuples) and a new point,
    try to extend some lines. Returns new lines if valid, else None."""
    # This is complex; we'll use a different approach.
    return None

def _random_valid_16():
    """Generate a random 16-point set with no 5 collinear, bounded coords."""
    while True:
        pts = []
        seen = set()
        for _ in range(16):
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            pts.append((x, y))
        # check no 5 collinear
        lines_map = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                k = _line_key(pts[i], pts[j])
                if k is not None:
                    lines_map[k].add(i)
        ok = True
        for k, idxs in lines_map.items():
            if len(idxs) >= 5:
                ok = False
                break
        if ok:
            return pts

def _local_search(pts, max_iters=2000):
    """Hill-climb: move one point to a new integer location to improve score."""
    pts = [list(p) for p in pts]
    best_score, _ = _score_points([tuple(p) for p in pts])
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # try moving each point to nearby locations
        for i in range(16):
            orig = tuple(pts[i])
            # candidate moves: small perturbations
            candidates = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = orig[0] + dx
                    ny = orig[1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    candidates.append((nx, ny))
            # also try some larger random jumps
            for _ in range(5):
                nx = random.randint(-1000, 1000)
                ny = random.randint(-1000, 1000)
                candidates.append((nx, ny))
            for cand in candidates:
                if cand == orig:
                    continue
                # check distinctness
                if any(cand == tuple(pts[j]) for j in range(16) if j != i):
                    continue
                new_pts = [tuple(p) for p in pts]
                new_pts[i] = cand
                s, _ = _score_points(new_pts)
                if s > best_score:
                    best_score = s
                    pts[i] = list(cand)
                    improved = True
                    break
            if improved:
                break
    return [tuple(p) for p in pts], best_score

def _line_crossover(parent1, parent2):
    """Crossover based on 4-point lines. Take lines from both parents, merge,
    then reconstruct a 16-point set from the merged line structure."""
    lines1 = _extract_lines(parent1)
    lines2 = _extract_lines(parent2)
    if lines1 is None or lines2 is None:
        return None
    # collect all lines (as frozensets of points)
    all_lines = []
    seen_lines = set()
    for ln in lines1 + lines2:
        fs = frozenset(ln)
        if fs not in seen_lines:
            seen_lines.add(fs)
            all_lines.append(ln)
    # Now we need to pick a subset of these lines such that the union of points
    # is exactly 16 distinct points and no 5 are collinear.
    # Greedy: pick lines that introduce new points, up to 16 points total.
    chosen = []
    used_pts = set()
    # sort lines by how many new points they add (prefer lines with more overlap? 
    # Actually we want lines that share points to be compatible)
    # Try to pick lines that are mutually consistent
    random.shuffle(all_lines)
    for ln in all_lines:
        ln_set = set(ln)
        # check: does adding this line keep total points <= 16 and no 5 collinear?
        new_used = used_pts | ln_set
        if len(new_used) > 16:
            continue
        # check no 5 collinear in the union
        # build a quick collinearity check on new_used
        pts_list = list(new_used)
        lines_map = defaultdict(set)
        for a in range(len(pts_list)):
            for b in range(a + 1, len(pts_list)):
                k = _line_key(pts_list[a], pts_list[b])
                if k is not None:
                    lines_map[k].add(a)
        ok = True
        for k, idxs in lines_map.items():
            if len(idxs) >= 5:
                ok = False
                break
        if not ok:
            continue
        chosen.append(ln)
        used_pts = new_used
        if len(used_pts) == 16:
            break
    if len(used_pts) != 16:
        # fill in with random points
        while len(used_pts) < 16:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            if (x, y) not in used_pts:
                used_pts.add((x, y))
    result = list(used_pts)
    # verify
    if len(set(result)) != 16:
        return None
    s, _ = _score_points(result)
    if s == 0:
        # might be invalid due to 5 collinear
        lines_map = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                k = _line_key(result[i], result[j])
                if k is not None:
                    lines_map[k].add(i)
        for k, idxs in lines_map.items():
            if len(idxs) >= 5:
                return None
    return result

def _mutate_points(pts, strength=0.3):
    """Randomly perturb some points."""
    pts = [list(p) for p in pts]
    n_mutate = random.randint(1, 5)
    for _ in range(n_mutate):
        i = random.randint(0, 15)
        if random.random() < 0.5:
            # small move
            pts[i][0] += random.randint(-5, 5)
            pts[i][1] += random.randint(-5, 5)
        else:
            # large move
            pts[i][0] = random.randint(-1000, 1000)
            pts[i][1] = random.randint(-1000, 1000)
        pts[i][0] = max(-1000, min(1000, pts[i][0]))
        pts[i][1] = max(-1000, min(1000, pts[i][1]))
    # ensure distinct
    result = []
    seen = set()
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            result.append(tuple(p))
    while len(result) < 16:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        if (x, y) not in seen:
            seen.add((x, y))
            result.append((x, y))
    return result

def _initial_grid():
    """The 4x4 grid baseline."""
    return [(x, y) for x in range(4) for y in range(4)]

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(42)
    
    best_pts = _initial_grid()
    best_score, _ = _score_points(best_pts)
    
    # Population of candidate point sets
    pop = [best_pts]
    # Add some random initial candidates
    for _ in range(10):
        rp = _random_valid_16()
        pop.append(rp)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        # Evaluate population
        scored = []
        for pts in pop:
            s, _ = _score_points(pts)
            scored.append((s, pts))
        scored.sort(key=lambda x: -x[0])
        
        # Track best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_pts = scored[0][1]
        
        # Elitism: keep top few
        new_pop = [p for _, p in scored[:4]]
        
        # Crossover and mutation to fill population
        while len(new_pop) < 16:
            if time.time() > deadline:
                break
            # pick two parents
            p1 = scored[random.randint(0, min(5, len(scored)-1))][1]
            p2 = scored[random.randint(0, min(5, len(scored)-1))][1]
            
            if random.random() < 0.5:
                # line crossover
                child = _line_crossover(p1, p2)
                if child is None:
                    child = _mutate_points(p1)
            else:
                child = _mutate_points(p1)
            
            # quick local search on child
            if time.time() < deadline - 10:
                child, cs = _local_search(child, max_iters=200)
                if cs > best_score:
                    best_score = cs
                    best_pts = child
            new_pop.append(child)
        
        pop = new_pop
    
    # Final polish: deeper local search on the best
    if time.time() < deadline - 5:
        best_pts, best_score = _local_search(best_pts, max_iters=5000)
    
    return [[x, y] for x, y in best_pts]
# EVOLVE-BLOCK-END
