import time
import random
import itertools
import sys
from math import gcd

DEADLINE = 145.0  # safety margin under 150s

def _collinear(p, q, r):
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]) == 0

def _all_triples(points):
    """Return list of (i,j,k) indices that are collinear."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_3free(points):
    """Exact largest subset with no three collinear via branch and bound."""
    n = len(points)
    triples = _all_triples(points)
    # Build for each point the set of triples it's in, and adjacency: 
    # point a and b are "conflicting" if there exists c such that (a,b,c) collinear
    # Actually for 3-free: a subset S is valid iff no triple is fully in S.
    # We want max |S|.
    
    # Precompute: for each pair (i,j), the set of k such that (i,j,k) collinear
    pair_conflicts = {}
    for (i,j,k) in triples:
        for (a,b) in [(i,j),(i,k),(j,k)]:
            pair_conflicts.setdefault((a,b), set()).add(k)
            pair_conflicts.setdefault((b,a), set()).add(k)
    
    # B&B: state = (current_size, included_set, candidate_set)
    # Prune: if current_size + len(candidate_set) <= best, return
    # Also: if no triple can be formed from candidate_set alone, we can add all remaining
    # (i.e., candidate_set is itself 3-free)
    
    best = [0]
    
    def _is_3free_subset(indices):
        """Check if a set of indices has no 3 collinear."""
        s = set(indices)
        for (i,j,k) in triples:
            if i in s and j in s and k in s:
                return False
        return True
    
    def _branch(included, candidates):
        if not candidates:
            if len(included) > best[0]:
                best[0] = len(included)
            return
        # Upper bound
        if len(included) + len(candidates) <= best[0]:
            return
        # If candidates themselves are 3-free, we can take all
        if _is_3free_subset(list(candidates)):
            val = len(included) + len(candidates)
            if val > best[0]:
                best[0] = val
            return
        # Pick a point to branch on: choose the one in most triples within candidates
        # For speed, just pick first
        p = next(iter(candidates))
        rest = candidates - {p}
        
        # Branch 1: include p
        # Check: is included + {p} still valid? i.e., no triple fully in included∪{p}
        # A triple is in included∪{p} if it's in included (impossible, included is valid) 
        # or it involves p and two from included
        valid = True
        for (i,j,k) in triples:
            if p in (i,j,k):
                other_two = [x for x in (i,j,k) if x != p]
                if other_two[0] in included and other_two[1] in included:
                    valid = False
                    break
        if valid:
            new_included = included | {p}
            # New candidates: remove any point that with two from new_included forms a triple
            # Actually we just recurse; the _is_3free check and branching handle it
            _branch(new_included, rest)
        
        # Branch 2: exclude p
        _branch(included, rest)
    
    _branch(set(), set(range(n)))
    return best[0]

def _valid_config(points):
    """Check 20 distinct points, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for (x,y) in points:
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x,y) in seen:
            return False
        seen.add((x,y))
    # Check no four collinear
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _collinear(points[i], points[j], points[k]) and \
                       _collinear(points[i], points[j], points[l]):
                        return False
    return True

def _grid_candidates():
    """Generate diverse starting configurations."""
    cands = []
    # Random points
    for _ in range(30):
        pts = set()
        while len(pts) < 20:
            pts.add((random.randint(-50,50), random.randint(-50,50)))
        cands.append(list(pts))
    # Structured: grid-like with perturbations
    for base in [(-10,-10,5,5), (-15,-15,6,6), (-20,-20,7,7)]:
        x0,y0,rows,cols = base
        for dx in range(0,3):
            for dy in range(0,3):
                pts = []
                for i in range(rows):
                    for j in range(cols):
                        if len(pts) < 20:
                            pts.append((x0 + j*2 + dx, y0 + i*2 + dy))
                if len(pts) >= 20:
                    cands.append(pts[:20])
    # Parabola perturbed
    for shift in range(-3,4):
        pts = [(x+shift, x*x) for x in range(-10,10)]
        # Check bounds
        if all(abs(x)<=100 and abs(y)<=100 for x,y in pts):
            cands.append(pts)
    return cands

def construct():
    start_time = time.time()
    
    # Generate candidates and find best
    candidates = _grid_candidates()
    best_score = 20
    best_config = None
    
    for pts in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
        if not _valid_config(pts):
            continue
        score = _max_3free(pts)
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            if best_score <= 10:
                break
    
    if best_config is None:
        # Fallback: parabola
        return [[x, x*x] for x in range(-10, 10)]
    
    # Local search: try moving one point to improve score
    # Given time constraints, do limited local search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        current = [list(p) for p in best_config]
        cur_score = best_score
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 5:
            improved = False
            # Try moving each point to a random nearby location
            for idx in range(20):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                orig = current[idx]
                for _ in range(20):
                    nx = orig[0] + random.randint(-5,5)
                    ny = orig[1] + random.randint(-5,5)
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if any(current[i] == (nx,ny) for i in range(20) if i != idx):
                        continue
                    trial = [list(p) for p in current]
                    trial[idx] = [nx, ny]
                    if not _valid_config(trial):
                        continue
                    s = _max_3free(trial)
                    if s < cur_score:
                        current = trial
                        cur_score = s
                        improved = True
                        break
        best_config = current
        best_score = cur_score
    
    return best_config
