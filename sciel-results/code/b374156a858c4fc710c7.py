# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # Triangular lattice: point = a + b*omega, omega = exp(i*pi/3)
    # Squared distance between (a1,b1) and (a2,b2):
    #   da^2 + da*db + db^2
    # Unit-distance (edge) offsets (dist_sq == 1):
    EDGES = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges(pts):
        e = 0
        n = len(pts)
        for i in range(n):
            pi = pts[i]
            for j in range(i + 1, n):
                if dist_sq(pi, pts[j]) == 1:
                    e += 1
        return e

    def is_valid(pts):
        if len(pts) != 32:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        for i in range(32):
            pi = pts[i]
            for j in range(i + 1, 32):
                if dist_sq(pi, pts[j]) < 1:
                    return False
        return True

    # ---- Build a hexagonal (concentric ring) seed of 32 points ----
    def ring_points(r):
        if r == 0:
            return [(0, 0)]
        pts = []
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        pos = (r, 0)
        for d in dirs:
            for _ in range(r):
                pts.append(pos)
                pos = (pos[0] + d[0], pos[1] + d[1])
        return pts

    def build_seed():
        points = [(0, 0)]
        for r in range(1, 6):
            for p in ring_points(r):
                if len(points) < 32:
                    points.append(p)
                else:
                    break
            if len(points) >= 32:
                break
        return points[:32]

    # ---- Local search: single-point moves to empty, valid cells ----
    def local_search(pts, max_time):
        pts = list(pts)
        best = count_edges(pts)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                # candidate cells: 6 neighbors + a few random nearby cells
                candidates = []
                for da, db in EDGES:
                    c = (old[0] + da, old[1] + db)
                    if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                        candidates.append(c)
                for _ in range(4):
                    da = random.randint(-3, 3)
                    db = random.randint(-3, 3)
                    c = (old[0] + da, old[1] + db)
                    if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                        candidates.append(c)

                for cand in candidates:
                    if cand in pts:
                        continue
                    valid = True
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(cand, pts[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    old_e = sum(1 for j in range(32) if j != i and dist_sq(old, pts[j]) == 1)
                    new_e = sum(1 for j in range(32) if j != i and dist_sq(cand, pts[j]) == 1)
                    delta = new_e - old_e
                    if delta > 0:
                        pts[i] = cand
                        best += delta
                        improved = True
                        break
        return pts, best

    # ---- Simulated annealing: single-point moves, Metropolis acceptance ----
    def anneal(pts, max_time):
        pts = list(pts)
        cur_e = count_edges(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 1.5
        T_min = 1e-4
        n_iter = 0
        while time.time() < max_time:
            n_iter += 1
            i = random.randrange(32)
            old = pts[i]
            # pick a random candidate cell near old
            for _ in range(8):
                da = random.randint(-3, 3)
                db = random.randint(-3, 3)
                cand = (old[0] + da, old[1] + db)
                if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                    continue
                if cand in pts:
                    continue
                valid = True
                for j in range(32):
                    if j == i:
                        continue
                    if dist_sq(cand, pts[j]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                old_e = sum(1 for j in range(32) if j != i and dist_sq(old, pts[j]) == 1)
                new_e = sum(1 for j in range(32) if j != i and dist_sq(cand, pts[j]) == 1)
                delta = new_e - old_e
                if delta >= 0 or random.random() < _boltzmann(delta, T):
                    pts[i] = cand
                    cur_e += delta
                    if cur_e > best_e:
                        best_e = cur_e
                        best_pts = list(pts)
                break
            # cool
            T = max(T_min, T * 0.9995)
        return best_pts, best_e

    def _boltzmann(delta, T):
        # exp(delta/T) with delta <= 0
        if delta == 0:
            return 1.0
        import math
        x = delta / T
        if x < -700:
            return 0.0
        return math.exp(x)

    # ---- Main ----
    seed = build_seed()
    if not is_valid(seed):
        return [[a, b] for a in range(16) for b in range(2)]

    best_pts, best_e = local_search(seed, deadline - 10.0)

    # Multiple annealing restarts from the locally-improved point
    while time.time() < deadline - 5.0:
        cand_pts, cand_e = anneal(best_pts, deadline - 5.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            # polish the new best with a short local search
            polished, pe = local_search(best_pts, deadline - 5.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
        # occasionally re-seed from the fresh hexagonal seed for diversity
        if random.random() < 0.3:
            cand_pts, cand_e = anneal(build_seed(), deadline - 5.0)
            if cand_e > best_e:
                best_pts, cand_e = local_search(cand_pts, deadline - 5.0)
                best_pts, best_e = cand_pts, cand_e

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
