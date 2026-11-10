# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points maximizing k
    subject to: min pairwise squared distance >= max squared norm.
    Uses mixed-norm candidate pool (norm^2 in {2, 4}) with greedy + local improvement.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    d = 11

    # Build candidate pool: all integer vectors with squared norm 2 or 4
    # Norm^2 = 2: exactly two coordinates are +/-1, rest 0 -> C(11,2)*4 = 220
    # Norm^2 = 4: patterns: (2,0,..) -> 11*4=44; (1,1,0,..) -> C(11,2)*4=220; (0,..) no
    #   Actually norm^2=4: one coord is +/-2 (11*4=44), or two coords are +/-1 (C(11,2)*4=220)
    #   Total norm^2=4: 44 + 220 = 264
    # Total: 220 + 264 = 484 candidates

    candidates = []
    # Norm^2 = 2: two coords are +/-1
    for i in range(d):
        for j in range(i+1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    # Norm^2 = 4: one coord is +/-2
    for i in range(d):
        for s in (2, -2):
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)

    # Norm^2 = 4: two coords are +/-1 (same as norm^2=2 pattern but we already have those)
    # Wait - norm^2 = 1+1 = 2, not 4. Let me reconsider.
    # Norm^2 = 4 patterns:
    #   - One coordinate is 2 or -2: sum = 4. Count: 11 * 2 = 22... wait 11*4=44? No, 11 positions * 2 signs = 22
    #   Actually: for each of 11 positions, value can be +2 or -2, so 11*2 = 22
    #   - Two coordinates with values that sum squares to 4: (2,0) already counted, (1,1) gives 1+1=2 not 4
    #   - Wait: 1^2+1^2 = 2, not 4. So norm^2=4 only from single coord +/-2? That's 22.
    #   Hmm, let me recheck: what about (1,1,1,1,0,...)? 1+1+1+1=4. Yes!
    #   Four coords each +/-1: C(11,4)*16 = 330*16 = 5280. That's a lot.
    #   
    # Let me be more careful. Norm^2 = 4:
    #   Pattern A: one coord = +/-2, rest 0. Count: 11*2 = 22
    #   Pattern B: two coords = +/-1... 1+1=2, not 4. NO.
    #   Pattern C: four coords = +/-1, rest 0. 1+1+1+1=4. Count: C(11,4)*2^4 = 330*16 = 5280
    #   Pattern D: one coord=+/-1, one=+/-sqrt(3)... no, must be integer.
    #   So norm^2=4: 22 + 5280 = 5302. That's large but manageable.
    #
    # Actually wait, I need to re-examine. The knowledge says "264 candidates" for norm^2=4.
    # Let me recount. Maybe they meant a restricted subset.
    # 
    # For norm^2=2: C(11,2)*4 = 220. For norm^2=4: if we only use single-coord +/-2: 22.
    # 220+22=242, not 264. If norm^2=4 includes two-coord patterns... 
    # 
    # Let me just use norm^2=2 (220) and norm^2=4 single-coord (22) = 242 total.
    # Or better: use the full norm^2=4 set but it's 5302 which is large.
    # 
    # Given the knowledge says 264 for norm^2=4, maybe they used a different definition.
    # Let me just use norm^2=2 (220) + norm^2=4 single-coord (22) = 242.
    # Actually, let me include norm^2=4 with two +/-1 coords... no that's norm^2=2.
    # 
    # I'll use: norm^2=2 (220 vectors) + norm^2=4 (22 vectors of type (±2,0,...,0)) = 242 total.
    # This matches a reasonable candidate pool.

    # Rebuild candidates properly
    candidates = []
    # Norm^2 = 2
    for i in range(d):
        for j in range(i+1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    # Norm^2 = 4: single coordinate +/-2
    for i in range(d):
        for s in (2, -2):
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)

    n = len(candidates)
    points_arr = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms2 = np.sum(points_arr**2, axis=1)
    
    # We need: for all pairs (i,j) in selected set, dist2(i,j) >= max_norm2
    # dist2(i,j) = norm2(i) + norm2(j) - 2*dot(i,j)
    # So we need: norm2(i) + norm2(j) - 2*dot(i,j) >= max_norm2
    # For max_norm2 = 4: need dot(i,j) <= (norm2(i)+norm2(j)-4)/2
    #   For (4,4): dot <= 2
    #   For (4,2): dot <= 1
    #   For (2,2): dot <= 0
    # For max_norm2 = 2: need dot(i,j) <= (norm2(i)+norm2(j)-2)/2
    #   For (2,2): dot <= 0
    #   For (4,2): dot <= 1  (but max would be 4, not 2)
    
    # Strategy: try with max_norm2 = 4 first (allows norm-4 points), fall back to max_norm2=2
    
    def check_pair_ok(i, j, max_norm2):
        """Check if points i and j can coexist given max_norm2."""
        ni = norms2[i]
        nj = norms2[j]
        # Need dist2 >= max_norm2
        # dist2 = ni + nj - 2*dot
        dot = int(np.dot(points_arr[i], points_arr[j]))
        dist2 = ni + nj - 2*dot
        return dist2 >= max_norm2

    # Precompute incompatibility for max_norm2 = 4
    # Two points are incompatible if dist2 < 4
    # For speed, precompute a conflict matrix
    
    def build_conflict(max_norm2):
        """Returns a set of (i,j) pairs that are incompatible."""
        conflicts = set()
        for i in range(n):
            for j in range(i+1, n):
                if not check_pair_ok(i, j, max_norm2):
                    conflicts.add((i, j))
        return conflicts

    def greedy_select(conflicts, max_norm2, time_limit=None):
        """Greedy: repeatedly pick the point with fewest conflicts among remaining."""
        selected = []
        remaining = set(range(n))
        # Only consider points whose norm2 <= max_norm2
        remaining = {i for i in remaining if norms2[i] <= max_norm2}
        
        while remaining:
            # Find point with minimum degree in conflict graph induced by remaining
            best = None
            best_deg = float('inf')
            for i in remaining:
                deg = 0
                for j in remaining:
                    if j == i:
                        continue
                    if (min(i,j), max(i,j)) in conflicts:
                        deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best = i
                if best_deg == 0:
                    break
            selected.append(best)
            remaining.discard(best)
            # Remove conflicts of best from consideration
            for j in list(remaining):
                if (min(best,j), max(best,j)) in conflicts:
                    remaining.discard(j)
            if time_limit and time.time() > time_limit:
                break
        return selected

    def local_improve(selected, conflicts, max_norm2, max_rounds=200, time_limit=None):
        """Try to add unselected points or swap to improve."""
        selected = list(selected)
        sel_set = set(selected)
        
        # Try adding any unselected point that doesn't conflict with selected
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if i in sel_set or norms2[i] > max_norm2:
                    continue
                # Check if i conflicts with any selected
                ok = True
                for j in selected:
                    if (min(i,j), max(i,j)) in conflicts:
                        ok = False
                        break
                if ok:
                    selected.append(i)
                    sel_set.add(i)
                    improved = True
                    if time_limit and time.time() > time_limit:
                        return selected
        
        # Swap: remove one selected, add two unselected? Or remove one, add one.
        # Try: for each selected point s, for each unselected point u, 
        # if removing s allows adding u (and u doesn't conflict with rest of selected)
        for _ in range(max_rounds):
            if time_limit and time.time() > time_limit:
                break
            found_swap = False
            for s in list(selected):
                rest = [x for x in selected if x != s]
                rest_set = set(rest)
                # Find unselected u that doesn't conflict with rest
                for u in range(n):
                    if u in rest_set or norms2[u] > max_norm2:
                        continue
                    ok = True
                    for r in rest:
                        if (min(u,r), max(u,r)) in conflicts:
                            ok = False
                            break
                    if ok:
                        # Swap s with u: size stays same, but might enable future additions
                        # Only do swap if it helps (e.g., u has fewer conflicts)
                        # For now, just try
                        selected = rest + [u]
                        sel_set = set(selected)
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break
        return selected

    best_result = None
    best_k = 0

    # Try max_norm2 = 4
    if time.time() < DEADLINE - 30:
        conflicts4 = build_conflict(4)
        sel4 = greedy_select(conflicts4, 4, time_limit=DEADLINE - 60)
        sel4 = local_improve(sel4, conflicts4, 4, max_rounds=100, time_limit=DEADLINE - 30)
        k4 = len(sel4)
        if k4 > best_k:
            best_k = k4
            best_result = sel4
        # Try multiple greedy restarts with different orderings
        restarts = 0
        while time.time() < DEADLINE - 60 and restarts < 50:
            # Randomize by shuffling
            np.random.seed(restarts)
            perm = np.random.permutation(n)
            # Greedy with random tie-breaking
            selected_r = []
            remaining = set(i for i in range(n) if norms2[i] <= 4)
            while remaining and time.time() < DEADLINE - 60:
                best = None
                best_deg = float('inf')
                for i in perm:
                    if i not in remaining:
                        continue
                    deg = 0
                    for j in remaining:
                        if j == i:
                            continue
                        if (min(i,j), max(i,j)) in conflicts4:
                            deg += 1
                    if deg < best_deg:
                        best_deg = deg
                        best = i
                    if best_deg == 0:
                        break
                if best is None:
                    break
                selected_r.append(best)
                remaining.discard(best)
                for j in list(remaining):
                    if (min(best,j), max(best,j)) in conflicts4:
                        remaining.discard(j)
            selected_r = local_improve(selected_r, conflicts4, 4, max_rounds=50, time_limit=DEADLINE - 30)
            if len(selected_r) > best_k:
                best_k = len(selected_r)
                best_result = selected_r
            restarts += 1

    # Try max_norm2 = 2 (only norm-2 points, need dot <= 0)
    if time.time() < DEADLINE - 30:
        conflicts2 = build_conflict(2)
        sel2 = greedy_select(conflicts2, 2, time_limit=DEADLINE - 60)
        sel2 = local_improve(sel2, conflicts2, 2, max_rounds=100, time_limit=DEADLINE - 30)
        if len(sel2) > best_k:
            best_k = len(sel2)
            best_result = sel2
        restarts = 0
        while time.time() < DEADLINE - 60 and restarts < 50:
            np.random.seed(1000 + restarts)
            perm = np.random.permutation(n)
            selected_r = []
            remaining = set(i for i in range(n) if norms2[i] <= 2)
            while remaining and time.time() < DEADLINE - 60:
                best = None
                best_deg = float('inf')
                for i in perm:
                    if i not in remaining:
                        continue
                    deg = 0
                    for j in remaining:
                        if j == i:
                            continue
                        if (min(i,j), max(i,j)) in conflicts2:
                            deg += 1
                    if deg < best_deg:
                        best_deg = deg
                        best = i
                    if best_deg == 0:
                        break
                if best is None:
                    break
                selected_r.append(best)
                remaining.discard(best)
                for j in list(remaining):
                    if (min(best,j), max(best,j)) in conflicts2:
                        remaining.discard(j)
            selected_r = local_improve(selected_r, conflicts2, 2, max_rounds=50, time_limit=DEADLINE - 30)
            if len(selected_r) > best_k:
                best_k = len(selected_r)
                best_result = selected_r
            restarts += 1

    # Fallback: always have at least the 2-point solution
    if best_result is None or len(best_result) < 2:
        best_result = [0, 1]  # first two norm-2 points (should be compatible)

    # Verify and return
    result = np.array([points_arr[i] for i in best_result], dtype=np.int64)
    result = np.around(result).astype(np.int64)
    
    # Final validation
    k = result.shape[0]
    if k < 2:
        result = np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
