import itertools
import random
import time

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def is_isosceles(a, b, c):
    d1 = dist2(a, b)
    d2 = dist2(a, c)
    d3 = dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def is_isosceles_free(points):
    pts = list(points)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                if is_isosceles(pts[i], pts[j], pts[k]):
                    return False
    return True

def is_maximal(points_set):
    # For every point not in points_set, there must be a,b in points_set
    # such that (p, a, b) is isosceles.
    pts = list(points_set)
    n = len(pts)
    for p in ALL_POINTS:
        if p in points_set:
            continue
        blocked = False
        for i in range(n):
            for j in range(i+1, n):
                if is_isosceles(p, pts[i], pts[j]):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def greedy_random_order(rng):
    chosen = []
    order = ALL_POINTS[:]
    rng.shuffle(order)
    for p in order:
        if all(not is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return set(chosen)

def greedy_row_major():
    chosen = []
    for p in ALL_POINTS:
        if all(not is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return set(chosen)

def construct():
    deadline = time.time() + 280  # safety margin
    rng = random.Random(42)
    
    # Start with greedy row-major
    best_set = greedy_row_major()
    best_size = len(best_set)
    
    # Try to improve by removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pts = list(best_set)
        rng.shuffle(pts)
        for p in pts:
            if time.time() > deadline:
                break
            candidate = best_set - {p}
            if not candidate:
                continue
            # Removing preserves isosceles-free
            if is_maximal(candidate):
                best_set = candidate
                best_size = len(best_set)
                improved = True
                break
    
    # Now try swaps: remove one, add another
    for _ in range(50000):
        if time.time() > deadline:
            break
        if len(best_set) < 2:
            break
        pts = list(best_set)
        rng.shuffle(pts)
        p_remove = pts[0]
        # Try to find a point to add
        candidates_add = [q for q in ALL_POINTS if q not in best_set]
        rng.shuffle(candidates_add)
        for q in candidates_add:
            if time.time() > deadline:
                break
            # Check if adding q keeps isosceles-free
            temp = best_set | {q}
            # Check isosceles-free: only need to check triples involving q
            ok = True
            pts_list = list(best_set)
            for i in range(len(pts_list)):
                for j in range(i+1, len(pts_list)):
                    if is_isosceles(q, pts_list[i], pts_list[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Now try removing p_remove from temp
            candidate = temp - {p_remove}
            if is_maximal(candidate):
                best_set = candidate
                best_size = len(best_set)
                break
    
    # Try adding points then removing different points (more complex swaps)
    for _ in range(20000):
        if time.time() > deadline:
            break
        if len(best_set) < 2:
            break
        # Add a random point
        candidates_add = [q for q in ALL_POINTS if q not in best_set]
        if not candidates_add:
            break
        q = rng.choice(candidates_add)
        temp = best_set | {q}
        # Check isosceles-free
        ok = True
        pts_list = list(best_set)
        for i in range(len(pts_list)):
            for j in range(i+1, len(pts_list)):
                if is_isosceles(q, pts_list[i], pts_list[j]):
                    ok = False
                    break
                if not ok:
                    break
        if not ok:
            continue
        # Now try to remove a random point from temp
        pts_temp = list(temp)
        rng.shuffle(pts_temp)
        for p_remove in pts_temp:
            candidate = temp - {p_remove}
            if is_maximal(candidate):
                best_set = candidate
                best_size = len(best_set)
                break
    
    return list(best_set)
