"""Search for integral-distance point sets using circle parametrization."""
from fractions import Fraction
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _is_squarefree(n):
    if n <= 1:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True


def _dist2(pts_i, pts_j, D):
    """Squared distance as Fraction."""
    dx = pts_i[0] - pts_j[0]
    dy = pts_i[1] - pts_j[1]
    return dx * dx + D * dy * dy


def _is_rational_dist(pts_i, pts_j, D):
    """Check if squared distance is a perfect square rational."""
    d2 = _dist2(pts_i, pts_j, D)
    num = d2.numerator
    den = d2.denominator
    if num == 0:
        return False
    return _is_perfect_square(num) and _is_perfect_square(den)


def _check_pairwise(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_dist(pts[i], pts[j], D):
                return False
    return True


def _check_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                   (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                    return False
    return True


def _check_concyclic(pts, D):
    """Check no four points lie on a conic x^2+D*y^2+Ax+By+C=0."""
    n = len(pts)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1 * x1 + D * y1 * y1
                    a2 = x2 * x2 + D * y2 * y2
                    a3 = x3 * x3 + D * y3 * y3
                    a4 = x4 * x4 + D * y4 * y4
                    det = (x1 * (y2 * (a3 - a4) - y3 * (a2 - a4) + y4 * (a2 - a3)) -
                           y1 * (x2 * (a3 - a4) - x3 * (a2 - a4) + x4 * (a2 - a3)) +
                           a1 * (x2 * y3 - x3 * y2 - x2 * y4 + x4 * y2 + x3 * y4 - x4 * y3) -
                           (x2 * y3 * a4 - x2 * y4 * a3 - x3 * y2 * a4 + x3 * y4 * a2 +
                            x4 * y2 * a3 - x4 * y3 * a2))
                    if det == 0:
                        return False
    return True


def _check_valid(pts, D):
    n = len(pts)
    if n < 3:
        return False
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    if not _check_pairwise(pts, D):
        return False
    if not _check_collinear(pts):
        return False
    if n >= 4 and not _check_concyclic(pts, D):
        return False
    return True


def _circle_points(limit_m, limit_n):
    """Generate points on unit circle via (m^2-n^2)/(m^2+n^2), 2mn/(m^2+n^2)."""
    pts = set()
    for m in range(1, limit_m + 1):
        for n in range(0, m):
            d = m * m + n * n
            x = Fraction(m * m - n * n, d)
            y = Fraction(2 * m * n, d)
            pts.add((x, y))
    return list(pts)


def _generate_candidates(D, limit):
    """Generate candidate points in [-limit, limit]^2 with integer coords."""
    pts = []
    for x in range(-limit, limit + 1):
        for y in range(-limit, limit + 1):
            pts.append((Fraction(x), Fraction(y)))
    return pts


def _precompute_compat(candidates, D):
    """Precompute compatibility matrix: compat[i][j] = True if pts i,j have rational distance."""
    n = len(candidates)
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _is_rational_dist(candidates[i], candidates[j], D):
                compat[i][j] = True
                compat[j][i] = True
    return compat


def _check_partial_fast(current_idx, compat, D):
    """Check pairwise distances for a set of indices using precomputed compat."""
    n = len(current_idx)
    for a in range(n):
        for b in range(a + 1, n):
            if not compat[current_idx[a]][current_idx[b]]:
                return False
    return True


def _check_collinear_idx(current_idx, candidates):
    """Check collinearity for a set of indices."""
    n = len(current_idx)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                pi = candidates[current_idx[i]]
                pj = candidates[current_idx[j]]
                pk = candidates[current_idx[k]]
                if (pj[0] - pi[0]) * (pk[1] - pi[1]) == \
                   (pk[0] - pi[0]) * (pj[1] - pi[1]):
                    return False
    return True


def _backtrack_search(candidates, D, target_n, max_nodes=500000, deadline=None):
    """
    Backtracking search for target_n valid points from candidates.
    Uses precomputed compatibility and smart ordering.
    """
    n_cand = len(candidates)
    compat = _precompute_compat(candidates, D)
    
    # Order candidates: prefer those with high degree in compat graph
    degrees = [sum(compat[i]) for i in range(n_cand)]
    order = sorted(range(n_cand), key=lambda i: -degrees[i])
    
    # Remap: work with ordered indices
    # Actually let's just use the original indices but try them in order of degree
    best_found = [None]
    node_count = [0]
    
    def check_full_concyclic(idx_list):
        pts = [candidates[i] for i in idx_list]
        return _check_concyclic(pts, D)
    
    def backtrack(current, start_pos):
        if node_count[0] > max_nodes:
            return False
        if deadline is not None and time.time() > deadline:
            return False
        
        if len(current) == target_n:
            if check_full_concyclic(current):
                best_found[0] = list(current)
                return True
            return False
        
        remaining_needed = target_n - len(current)
        # Pruning: not enough candidates left
        if n_cand - start_pos < remaining_needed:
            return False
        
        for pos in range(start_pos, n_cand):
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            if deadline is not None and time.time() > deadline:
                return False
            
            idx = order[pos]
            
            # Check pairwise distance with all current
            ok = True
            for ci in current:
                if not compat[ci][idx]:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check collinearity
            new_current = current + [idx]
            if not _check_collinear_idx(new_current, candidates):
                continue
            
            if len(new_current) == target_n:
                if check_full_concyclic(new_current):
                    best_found[0] = list(new_current)
                    return True
            else:
                if backtrack(new_current, pos + 1):
                    return True
        
        return False
    
    # Try multiple starting triples to increase success rate
    # First, find a good starting triple
    best_triple = None
    best_triple_score = -1
    
    # Sample triples from high-degree candidates
    top_candidates = order[:min(50, n_cand)]
    for i in range(len(top_candidates)):
        for j in range(i + 1, len(top_candidates)):
            a, b = top_candidates[i], top_candidates[j]
            if not compat[a][b]:
                continue
            pa, pb = candidates[a], candidates[b]
            if (pb[0] - pa[0]) * 0 == 0 and (pb[1] - pa[1]) == 0:
                continue
            for k in range(j + 1, len(top_candidates)):
                c = top_candidates[k]
                if not compat[a][c] or not compat[b][c]:
                    continue
                pc = candidates[c]
                # Check collinearity
                if (pb[0] - pa[0]) * (pc[1] - pa[1]) == (pc[0] - pa[0]) * (pb[1] - pa[1]):
                    continue
                # Score: count compatible extensions
                score = 0
                for d in top_candidates:
                    if d == a or d == b or d == c:
                        continue
                    if compat[a][d] and compat[b][d] and compat[c][d]:
                        pd = candidates[d]
                        # Check not collinear with any pair
                        coll = False
                        for p1, p2 in [(pa, pb), (pa, pc), (pb, pc)]:
                            if (p2[0] - p1[0]) * (pd[1] - p1[1]) == (pd[0] - p1[0]) * (p2[1] - p1[1]):
                                coll = True
                                break
                        if not coll:
                            score += 1
                if score > best_triple_score:
                    best_triple_score = score
                    best_triple = [a, b, c]
    
    # Try backtracking from best triple
    if best_triple is not None:
        # Start from the triple
        node_count[0] = 0
        # Find positions in order for the triple
        triple_set = set(best_triple)
        start_positions = [i for i, idx in enumerate(order) if idx in triple_set]
        
        # Direct backtrack from triple
        if len(best_triple) < target_n:
            # Continue from triple
            def backtrack_from_triple(current, start_pos):
                if node_count[0] > max_nodes:
                    return False
                if deadline is not None and time.time() > deadline:
                    return False
                
                if len(current) == target_n:
                    if check_full_concyclic(current):
                        best_found[0] = list(current)
                        return True
                    return False
                
                for pos in range(start_pos, n_cand):
                    node_count[0] += 1
                    if node_count[0] > max_nodes:
                        return False
                    if deadline is not None and time.time() > deadline:
                        return False
                    
                    idx = order[pos]
                    if idx in set(current):
                        continue
                    
                    ok = True
                    for ci in current:
                        if not compat[ci][idx]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_current = current + [idx]
                    if not _check_collinear_idx(new_current, candidates):
                        continue
                    
                    if len(new_current) == target_n:
                        if check_full_concyclic(new_current):
                            best_found[0] = list(new_current)
                            return True
                    else:
                        if backtrack_from_triple(new_current, pos + 1):
                            return True
                return False
            
            # Find the max position in order for the triple
            max_pos = max(i for i, idx in enumerate(order) if idx in triple_set)
            backtrack_from_triple(list(best_triple), max_pos + 1)
    
    if best_found[0] is not None:
        return [candidates[i] for i in best_found[0]]
    
    # Fallback: full backtrack from empty
    node_count[0] = 0
    backtrack([], 0)
    if best_found[0] is not None:
        return [candidates[i] for i in best_found[0]]
    return None


def _greedy_build(candidates, D, max_size=12):
    """Greedy: add points one by one if valid."""
    pts = []
    for cand in candidates:
        new_pts = pts + [cand]
        if _check_valid(new_pts, D):
            pts = new_pts
            if len(pts) >= max_size:
                break
    return pts


def _local_extend(pts, D, candidates, max_iter=5000, deadline=None):
    """Try to extend pts by adding one compatible candidate."""
    best = list(pts)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        added = False
        for cand in candidates:
            if cand in best:
                continue
            new_pts = best + [cand]
            if _check_valid(new_pts, D):
                best = new_pts
                added = True
                break
        if not added:
            break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    def remaining():
        return deadline - time.time()
    
    # Fallback: known 5-point set with D=1
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    fallback_5 = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in fracs_5]
    best_result = {'D': 1, 'points': fallback_5, 'n': 5}
    
    # Strategy 1: Grid-based search with D=1 (most likely to work)
    for limit in [5, 6, 7, 8, 9, 10, 12, 15]:
        if remaining() < 15:
            break
        if limit * limit > 500:
            continue
        
        candidates = _generate_candidates(1, limit)
        # Remove origin to avoid trivial issues? No, keep it.
        
        # Try backtracking for 6 points
        if remaining() > 30:
            result = _backtrack_search(candidates, 1, 6, max_nodes=300000, deadline=deadline - 10)
            if result:
                pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]
                best_result = {'D': 1, 'points': pts_out, 'n': len(result)}
                
                # Try to extend
                if remaining() > 20:
                    extended = _local_extend(result, 1, candidates, max_iter=3000, deadline=deadline - 5)
                    if len(extended) > best_result['n']:
                        pts_out2 = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in extended]
                        best_result = {'D': 1, 'points': pts_out2, 'n': len(extended)}
        
        # Try for 7 points if we have time
        if remaining() > 40 and best_result['n'] >= 6:
            result7 = _backtrack_search(candidates, 1, 7, max_nodes=200000, deadline=deadline - 10)
            if result7:
                pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result7]
                best_result = {'D': 1, 'points': pts_out, 'n': len(result7)}
        
        if best_result['n'] >= 7:
            break
    
    # Strategy 2: Circle-based candidates with D=1
    if remaining() > 20:
        # Points on unit circle + origin
        circle_pts = _circle_points(12, 12)
        candidates_c = [(Fraction(0), Fraction(0))] + circle_pts
        # Deduplicate
        seen = set()
        unique_cands = []
        for p in candidates_c:
            if p not in seen:
                seen.add(p)
                unique_cands.append(p)
        
        if remaining() > 25:
            result = _backtrack_search(unique_cands, 1, 6, max_nodes=200000, deadline=deadline - 10)
            if result and len(result) > best_result['n']:
                pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]
                best_result = {'D': 1, 'points': pts_out, 'n': len(result)}
    
    # Strategy 3: Try D=2 with grid
    if remaining() > 20:
        for limit in [5, 6, 7, 8]:
            if remaining() < 15:
                break
            candidates = _generate_candidates(2, limit)
            if remaining() > 20:
                result = _backtrack_search(candidates, 2, 6, max_nodes=200000, deadline=deadline - 10)
                if result and len(result) > best_result['n']:
                    pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]
                    best_result = {'D': 2, 'points': pts_out, 'n': len(result)}
            if best_result['n'] >= 6:
                break
    
    # Final validation
    final_pts = [ (Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) for p in best_result['points'] ]
    D_final = best_result['D']
    if not _check_valid(final_pts, D_final):
        # Use fallback
        best_result = {'D': 1, 'points': fallback_5, 'n': 5}
    
    del best_result['n']
    return best_result

# EVOLVE-BLOCK-END
