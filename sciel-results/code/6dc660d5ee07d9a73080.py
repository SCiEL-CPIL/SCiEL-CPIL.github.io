# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 30.0


def _build_tables():
    """Build field arithmetic, line membership, and the baseline Nikodym set."""
    p = _P
    k = 3  # alpha^2 = 3 (3 is a quadratic non-residue mod 7)

    def pt_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p * p + ya * p + yb)

    # Build line_points: list of arrays of point indices
    # Line types:
    #   0..48: vertical lines (fixed x, all y)
    #   49..2448: slanted lines (slope m, offset (oa,ob))
    line_points = []

    # Vertical lines: for each x in F_49, the line { (x, y) : y in F_49 }
    for xa in range(p):
        for xb in range(p):
            pts = []
            for t in range(p * p):
                ta, tb = t // p, t % p
                ya, yb = ta, tb
                pts.append(pt_idx(xa, xb, ya, yb))
            line_points.append(np.array(pts, dtype=np.int64))

    # Slanted lines: for each slope m in F_49 and offset (oa,ob) in F_49^2
    # Line: { (t, m*t + offset) : t in F_49 }
    for m in range(p * p):
        ma, mb = m // p, m % p
        for oa in range(p):
            for ob in range(p):
                pts = []
                for t in range(p * p):
                    ta, tb = t // p, t % p
                    # m*t = (ma+mb*alpha)*(ta+tb*alpha) = (ma*ta + mb*tb*3, ma*tb + mb*ta)
                    ya = (ma * ta + mb * tb * k + oa) % p
                    yb = (ma * tb + mb * ta + ob) % p
                    pts.append(pt_idx(ta, tb, ya, yb))
                line_points.append(np.array(pts, dtype=np.int64))

    n_lines = len(line_points)  # 49 + 49*49 = 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(_Q2)]
    for li, pts in enumerate(line_points):
        for pi in pts:
            point_lines[pi].append(li)

    # Baseline: S = { (x,y) : 2*xa + ya^2 - yb^2*3 == 0 (mod 7) AND ya NOT a square mod 7 }
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    S = np.zeros(_Q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S[pt_idx(xa, xb, ya, yb)] = True

    return p, k, line_points, point_lines, S


def nikodym_set() -> np.ndarray:
    p, k, line_points, point_lines, S = _build_tables()
    q = _Q
    q2 = _Q2
    n_lines = len(line_points)

    # Precompute line_S_count and cover_count
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    for li in range(n_lines):
        pts = line_points[li]
        cnt = 0
        for pi in pts:
            if S[pi]:
                cnt += 1
        line_S_count[li] = cnt

    # cover_count[x] = number of lines L containing x such that:
    #   if x in S: S_count(L) == 1 and x is the S-point on L
    #   if x not in S: S_count(L) == 0
    cover_count = np.zeros(q2, dtype=np.int64)
    for li in range(n_lines):
        pts = line_points[li]
        s_count = 0
        s_point = -1
        for pi in pts:
            if S[pi]:
                s_count += 1
                s_point = pi
                if s_count > 1:
                    break
        if s_count == 0:
            for pi in pts:
                cover_count[pi] += 1
        elif s_count == 1:
            cover_count[s_point] += 1

    # Verify baseline validity
    if np.any(cover_count == 0):
        # Should not happen with correct baseline; fallback to all points
        result = np.zeros((q2, 2, 2), dtype=np.int64)
        for z in range(q2):
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            result[z] = [[xa, xb], [ya, yb]]
        return result

    # Track zero-line count per point: number of lines through z with S_count == 0
    zero_line_count = np.zeros(q2, dtype=np.int64)
    for z in range(q2):
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                zero_line_count[z] += 1

    def calc_damage(z):
        """Calculate the weighted damage to other points if z is added to S."""
        if S[z]:
            return float('inf')
        if zero_line_count[z] == 0:
            return float('inf')
        
        damage = 0.0
        for li in point_lines[z]:
            old_count = line_S_count[li]
            if old_count == 0:
                # All other points on L lose a 0-line
                for pi in line_points[li]:
                    if pi != z:
                        # Weight by inverse cover count of pi
                        # If pi is in S, it needs 1-line, not 0-line.
                        # Wait, if pi in S, it is covered by 1-lines.
                        # Adding z to S makes L have count 1.
                        # Does L become a valid cover for pi?
                        # No, L now has z in S. If pi != z, L has 2 points in S? No, just z.
                        # So L is a 1-line.
                        # If pi in S, L is a valid cover for pi ONLY IF pi is the unique S point.
                        # But z is the unique S point. So L is NOT a valid cover for pi (if pi in S).
                        # So pi loses a 0-line (irrelevant for S points) and gains nothing?
                        # Actually, if pi in S, it was covered by 1-lines. L was 0-line, so not a cover.
                        # Now L is 1-line (with z). Still not a cover for pi.
                        # So no change for pi in S.
                        
                        # If pi not in S:
                        # It was covered by 0-lines. L was a 0-line.
                        # Now L is a 1-line. So pi loses this 0-line.
                        if not S[pi]:
                            cc = cover_count[pi]
                            if cc > 0:
                                damage += 1.0 / cc
                            else:
                                damage += 100.0 # Should not happen if valid
            elif old_count == 1:
                # The existing S point on L loses this 1-line?
                # L had 1 S point (say u).
                # Now L has 2 S points (u and z).
                # So L is no longer a 1-line.
                # u loses this cover.
                for pi in line_points[li]:
                    if S[pi] and pi != z:
                        cc = cover_count[pi]
                        if cc > 0:
                            damage += 1.0 / cc
                        else:
                            damage += 100.0
                        break
            # old_count >= 2: no change in coverage type (was invalid, still invalid)
        
        return damage

    # Priority queue based on Damage
    # We want to add points with MINIMUM damage.
    
    def build_pq():
        pq = []
        for z in range(q2):
            if S[z] or zero_line_count[z] == 0:
                continue
            d = calc_damage(z)
            if d < float('inf'):
                heapq.heappush(pq, (d, z))
        return pq

    def try_add(z):
        """Try adding z to S. Returns True if valid (and keeps the change)."""
        if zero_line_count[z] == 0:
            return False

        # Check that no other point loses its last covering line
        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            if old_count == 0:
                # L used to cover all its non-S points; now covers none (for non-S)
                # and covers z (if z was the only S point? No, z is added, so count becomes 1).
                # Wait, if old_count == 0, L is a 0-line.
                # After adding z, L is a 1-line (z is the S point).
                # Non-S points on L lose this 0-line.
                for pi in line_points[li]:
                    if pi != z and not S[pi]:
                        if cover_count[pi] <= 1:
                            return False
            elif old_count == 1:
                # L used to cover one S-point (u); now covers no one (count 2)
                for pi in line_points[li]:
                    if S[pi] and pi != z:
                        if cover_count[pi] <= 1:
                            return False
                        break
            # old_count >= 2: no change in coverage

        # All checks passed: commit the addition
        S[z] = True
        for li in affected_lines:
            old_count = line_S_count[li]
            line_S_count[li] += 1
            if old_count == 0:
                for pi in line_points[li]:
                    if pi != z and not S[pi]:
                        cover_count[pi] -= 1
                        zero_line_count[pi] -= 1
            elif old_count == 1:
                for pi in line_points[li]:
                    if S[pi] and pi != z:
                        cover_count[pi] -= 1
                        break

        # Recompute z's cover_count
        new_z_cover = 0
        for li in affected_lines:
            if line_S_count[li] == 1:
                for pi in line_points[li]:
                    if S[pi]:
                        if pi == z:
                            new_z_cover += 1
                        break
        cover_count[z] = new_z_cover
        return True

    # Main Loop: Min-Damage Greedy
    while time.time() < _DEADLINE - _MARGIN:
        pq = build_pq()
        if not pq:
            break
        
        added_any = False
        while pq:
            if time.time() > _DEADLINE - _MARGIN:
                break
            d, z = heapq.heappop(pq)
            if S[z] or zero_line_count[z] == 0:
                continue
            # Re-check damage? Optional, but safer.
            # If calc_damage is expensive, we might skip re-check.
            # But state changes.
            # Let's just try_add.
            if try_add(z):
                added_any = True
                # Invalidate PQ?
                # Damage of other points changed.
                # We break out and rebuild PQ.
                break
        
        if not added_any:
            # Try a random restart or different strategy?
            # For now, if greedy fails, try a few random valid adds?
            # If no valid adds exist, we are done.
            break

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    if not pts_list:
        # Fallback: return all points
        for z in range(q2):
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
