import time
import random
from collections import Counter

_DEADLINE = 90.0
_MARGIN = 8.0
_START = time.time()

def _max_mult(ys, xs=None):
    """Compute max multiplicity of a single positive doubled area for points.
    If xs is None, assume all x are 0 (single vertical line)."""
    n = len(ys)
    c = Counter()
    if xs is None:
        # All points on x=0: doubled area of (i,j,k) = |ys[j]-ys[k]| if same x... 
        # Actually all same x => all collinear => area 0. Not useful.
        return 0
    # General case with given xs and ys
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = abs((xs[j]-xs[i])*(ys[k]-ys[i]) - (xs[k]-xs[i])*(ys[j]-ys[i]))
                if a > 0:
                    c[a] += 1
    if not c:
        return 0
    return max(c.values())

def _max_mult_two_lines(yl, yr, xl=-1, xr=1):
    """Two parallel vertical lines. Points: (xl, y) for y in yl, (xr, y) for y in yr."""
    c = Counter()
    nl = len(yl)
    nr = len(yr)
    # Two left, one right: area = |xl - xr| * |(y1-y2)| = 1*|y1-y2| ... wait
    # det = (xr-xl)*(y2-y1) - ... let me just compute generally
    # Point i=(xl, a), j=(xl, b), k=(xr, c):
    # det = (xl-xl)*(c-a) - (xr-xl)*(b-a) = 0 - (xr-xl)*(b-a) = -(xr-xl)*(b-a)
    # abs = |xr-xl| * |b-a| = 1 * |b-a|
    for i in range(nl):
        for j in range(i+1, nl):
            d = abs(yl[j] - yl[i])
            if d > 0:
                c[d] += nr
    for i in range(nr):
        for j in range(i+1, nr):
            d = abs(yr[j] - yr[i])
            if d > 0:
                c[d] += nl
    # One left, two right:
    # Point i=(xl,a), j=(xr,b), k=(xr,c):
    # det = (xr-xl)*(c-a) - (xr-xl)*(b-a) = (xr-xl)*(c-b)
    # abs = |xr-xl|*|c-b| = |c-b|
    # Wait, that's same as two-right case. Let me recompute.
    # i=(xl,a), j=(xr,b), k=(xr,c)
    # (xj-xi)*(yk-yi) - (xk-xi)*(yj-yi) = (xr-xl)*(c-a) - (xr-xl)*(b-a) = (xr-xl)*((c-a)-(b-a)) = (xr-xl)*(c-b)
    # abs = |xr-xl| * |c-b| = |c-b|
    # Hmm, so one-left-two-right gives |c-b| which is the difference of the two right y's.
    # And two-left-one-right gives |b-a| which is the difference of the two left y's.
    # So actually:
    # - Two on same line: area = |diff on that line|, multiplicity = size of other line
    # - One on each: area = |diff between the two on the other line|... 
    # Wait no. Let me redo: two left (a,b) one right (c): area = |b-a| (the left pair's diff)
    # one left (a) two right (b,c): area = |c-b| (the right pair's diff)
    # So the area only depends on the pair on the SAME line. The third point's y doesn't matter!
    # That means: for each pair on left line with diff d, there are nr triangles with area d.
    # For each pair on right line with diff d, there are nl triangles with area d.
    # This is exactly what I coded above. Good.
    if not c:
        return 0
    return max(c.values())

def _generate_candidates():
    """Generate candidate configurations (yl, yr) for two-line setups."""
    candidates = []
    
    # Family 1: Two arithmetic progressions
    for d in range(1, 8):
        for start_l in range(-30, 31 - d*14):
            yl = [start_l + k*d for k in range(15)]
            if yl[-1] > 30 or yl[0] < -30:
                continue
            for start_r in range(-30, 31 - d*14):
                yr = [start_r + k*d for k in range(15)]
                if yr[-1] > 30 or yr[0] < -30:
                    continue
                candidates.append((yl, yr))
    
    # Family 2: Split a single AP into two groups
    for d in range(1, 6):
        for start in range(-30, 31 - d*29):
            base = [start + k*d for k in range(30)]
            if base[-1] > 30 or base[0] < -30:
                continue
            # Split into first 15 and last 15
            candidates.append((base[:15], base[15:]))
            # Interleaved split
            yl = base[0::2][:15]
            yr = base[1::2][:15]
            candidates.append((yl, yr))
    
    # Family 3: Two APs with different gaps
    for d1 in range(1, 6):
        for d2 in range(1, 6):
            for start_l in range(-30, 31 - d1*14):
                yl = [start_l + k*d1 for k in range(15)]
                if yl[-1] > 30 or yl[0] < -30:
                    continue
                for start_r in range(-30, 31 - d2*14):
                    yr = [start_r + k*d2 for k in range(15)]
                    if yr[-1] > 30 or yr[0] < -30:
                        continue
                    candidates.append((yl, yr))
    
    return candidates

def _local_search(yl, yr, steps=50):
    """Perturb y-values slightly and keep improvements."""
    yl = list(yl)
    yr = list(yr)
    best = _max_mult_two_lines(yl, yr)
    
    for _ in range(steps):
        if time.time() - _START > _DEADLINE - _MARGIN:
            break
        # Try perturbing one point
        improved = False
        for _attempt in range(20):
            if time.time() - _START > _DEADLINE - _MARGIN:
                break
            # Randomly pick a point to move
            if random.random() < 0.5:
                idx = random.randrange(len(yl))
                old = yl[idx]
                yl[idx] = old + random.choice([-2,-1,1,2])
                if yl[idx] < -30 or yl[idx] > 30:
                    yl[idx] = old
                    continue
                # Check distinctness
                if yl[idx] in yl[:idx] or yl[idx] in yl[idx+1:]:
                    yl[idx] = old
                    continue
                new_score = _max_mult_two_lines(yl, yr)
                if new_score > best:
                    best = new_score
                    improved = True
                else:
                    yl[idx] = old
            else:
                idx = random.randrange(len(yr))
                old = yr[idx]
                yr[idx] = old + random.choice([-2,-1,1,2])
                if yr[idx] < -30 or yr[idx] > 30:
                    yr[idx] = old
                    continue
                if yr[idx] in yr[:idx] or yr[idx] in yr[idx+1:]:
                    yr[idx] = old
                    continue
                new_score = _max_mult_two_lines(yl, yr)
                if new_score > best:
                    best = new_score
                    improved = True
                else:
                    yr[idx] = old
            if improved:
                break
    return yl, yr, best

def construct():
    global _START
    _START = time.time()
    
    # Seed the seed program's answer as baseline
    seed_pts = [(x, y) for x in range(6) for y in range(5)]
    seed_ys_l = [y for x, y in seed_pts if x in (0,1,2)]
    seed_ys_r = [y for x, y in seed_pts if x in (3,4,5)]
    # Actually the seed is a 6x5 grid, not two lines. Let me just compute its score.
    seed_score = 0
    c = Counter()
    n = len(seed_pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1,y1 = seed_pts[i]
                x2,y2 = seed_pts[j]
                x3,y3 = seed_pts[k]
                a = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if a > 0:
                    c[a] += 1
    if c:
        seed_score = max(c.values())
    
    best_score = seed_score
    best_pts = seed_pts
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    random.shuffle(candidates)
    
    for yl, yr in candidates:
        if time.time() - _START > _DEADLINE - _MARGIN:
            break
        # Check all 30 points are distinct and in range
        all_pts = [(-1, y) for y in yl] + [(1, y) for y in yr]
        if len(set(all_pts)) != 30:
            continue
        if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in all_pts):
            continue
        
        score = _max_mult_two_lines(yl, yr)
        if score > best_score:
            best_score = score
            best_pts = all_pts
            # Try local search
            yl2, yr2, ls_score = _local_search(yl, yr)
            if ls_score > best_score:
                best_score = ls_score
                best_pts = [(-1, y) for y in yl2] + [(1, y) for y in yr2]
    
    # Final check: ensure validity
    if len(set(best_pts)) != 30:
        return seed_pts
    if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in best_pts):
        return seed_pts
    
    return best_pts
