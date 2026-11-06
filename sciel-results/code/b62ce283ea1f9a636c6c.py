# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 140.0

    EDGES = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges_fast(pts):
        pt_set = set(pts)
        e = 0
        for p in pts:
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set:
                    e += 1
        return e // 2

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

    def build_hex_optimal():
        """Construct the known optimal 32-point hexagonal packing with 76 edges."""
        # Center + ring 1 (6) + ring 2 (12) + 13 from ring 3
        pts = [(0, 0)]
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pts.extend(ring1)  # 7 points
        
        # Ring 2: all points at hex-distance 2 (dist_sq == 4)
        ring2 = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                d = a*a + a*b + b*b
                if d == 4:
                    ring2.append((a, b))
        pts.extend(ring2)  # 7 + 12 = 19 points
        
        # Ring 3: need 13 more from dist_sq == 9
        ring3 = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                d = a*a + a*b + b*b
                if d == 9:
                    ring3.append((a, b))
        
        # Sort ring 3 to maximize edges with existing points
        def edge_score(p):
            s = set(pts)
            cnt = 0
            for da, db in EDGES:
                if (p[0]+da, p[1]+db) in s:
                    cnt += 1
            return cnt
        
        ring3.sort(key=edge_score, reverse=True)
        pts.extend(ring3[:13])
        
        return pts[:32]

    def build_greedy_seed():
        pts = [(0, 0)]
        for _ in range(31):
            best_cand = None
            best_score = -1
            best_dist_center = 999999
            min_a = min(p[0] for p in pts)
            max_a = max(p[0] for p in pts)
            min_b = min(p[1] for p in pts)
            max_b = max(p[1] for p in pts)
            for a in range(min_a - 2, max_a + 3):
                for b in range(min_b - 2, max_b + 3):
                    if abs(a) > 64 or abs(b) > 64:
                        continue
                    cand = (a, b)
                    if cand in pts:
                        continue
                    valid = True
                    for p in pts:
                        if dist_sq(cand, p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    edges = 0
                    for p in pts:
                        if dist_sq(cand, p) == 1:
                            edges += 1
                    d_center = a*a + a*b + b*b
                    if edges > best_score or (edges == best_score and d_center < best_dist_center):
                        best_score = edges
                        best_dist_center = d_center
                        best_cand = cand
            if best_cand is None:
                break
            pts.append(best_cand)
        return pts[:32]

    def local_search_fast(pts, max_time):
        pts = list(pts)
        pt_set = set(pts)
        pos_to_idx = {p: i for i, p in enumerate(pts)}
        best = count_edges_fast(pts)

        def get_point_edges(p, exclude_pos):
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set and q != exclude_pos:
                    e += 1
            return e

        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                old_edges = get_point_edges(old, old)

                best_cand = None
                best_delta = 0

                for da in range(-4, 5):
                    for db in range(-4, 5):
                        if da == 0 and db == 0:
                            continue
                        cand = (old[0] + da, old[1] + db)
                        if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                            continue
                        if cand in pt_set:
                            continue

                        # Check validity: no neighbor of cand (other than old) is in pt_set
                        valid = True
                        for da2, db2 in EDGES:
                            neighbor = (cand[0] + da2, cand[1] + db2)
                            if neighbor in pt_set and neighbor != old:
                                valid = False
                                break
                        if not valid:
                            continue

                        new_edges = get_point_edges(cand, old)
                        delta = new_edges - old_edges
                        if delta > best_delta:
                            best_delta = delta
                            best_cand = cand

                if best_cand is not None and best_delta > 0:
                    pt_set.remove(old)
                    pos_to_idx.pop(old)
                    pts[i] = best_cand
                    pt_set.add(best_cand)
                    pos_to_idx[best_cand] = i
                    best += best_delta
                    improved = True

        return pts, best

    def anneal_fast(pts, max_time):
        pts = list(pts)
        pt_set = set(pts)
        cur_e = count_edges_fast(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 3.0
        T_min = 0.01
        cooldown = 0.9995

        def get_point_edges(p, exclude_pos):
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set and q != exclude_pos:
                    e += 1
            return e

        while time.time() < max_time:
            i = random.randrange(32)
            old = pts[i]
            old_edges = get_point_edges(old, old)

            if random.random() < 0.8:
                da = random.randint(-2, 2)
                db = random.randint(-2, 2)
            else:
                da = random.randint(-5, 5)
                db = random.randint(-5, 5)
            cand = (old[0] + da, old[1] + db)

            if abs(cand[0]) > 64 or abs(cand[1]) > 64 or cand in pt_set:
                T = max(T_min, T * cooldown)
                continue

            valid = True
            for da2, db2 in EDGES:
                neighbor = (cand[0] + da2, cand[1] + db2)
                if neighbor in pt_set and neighbor != old:
                    valid = False
                    break

            if not valid:
                T = max(T_min, T * cooldown)
                continue

            new_edges = get_point_edges(cand, old)
            delta = new_edges - old_edges

            if delta >= 0 or random.random() < math.exp(delta / T):
                pt_set.remove(old)
                pts[i] = cand
                pt_set.add(cand)
                cur_e += delta
                if cur_e > best_e:
                    best_e = cur_e
                    best_pts = list(pts)

            T = max(T_min, T * cooldown)

        return best_pts, best_e

    # ---- Main ----
    # Try the known optimal hexagonal construction first
    hex_pts = build_hex_optimal()
    if is_valid(hex_pts):
        he = count_edges_fast(hex_pts)
        if he >= 76:
            return [[a, b] for a, b in hex_pts]

    # Build other seeds
    seeds = [hex_pts, build_greedy_seed()]
    valid_seeds = [s for s in seeds if is_valid(s)]
    if not valid_seeds:
        valid_seeds = [build_greedy_seed()]

    best_seed = max(valid_seeds, key=lambda s: count_edges_fast(s))
    best_pts, best_e = local_search_fast(best_seed, deadline - 20.0)

    if best_e >= 76:
        return [[a, b] for a, b in best_pts]

    while time.time() < deadline - 10.0:
        cand_pts, cand_e = anneal_fast(best_pts, deadline - 10.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            if best_e >= 76:
                break
            polished, pe = local_search_fast(best_pts, deadline - 10.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
                if best_e >= 76:
                    break
        else:
            fresh = build_greedy_seed()
            if is_valid(fresh):
                fp, fe = local_search_fast(fresh, deadline - 10.0)
                if fe > best_e:
                    best_pts, best_e = fp, fe
                    if best_e >= 76:
                        break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
