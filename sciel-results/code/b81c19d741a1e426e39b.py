"""
Minimum maximal no-isosceles set via hypergraph hitting set + precomputed bitmasks.

Strategy:
- Precompute, for each point, a bitmask of points that form an isosceles triangle with it and each other point.
- Use a greedy hitting-set approach: select points that cover (block) the most uncovered grid points.
- After greedy construction, perform local search (swap) to minimize size while maintaining maximality.
- Use precomputed distance tables and bitmask operations for speed.
"""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE = 320.0
TOTAL_POINTS = N * N  # 256


def _point_to_idx(p):
    return p[1] * N + p[0]


def _idx_to_point(idx):
    return (idx % N, idx // N)


def construct():
    start_time = time.time()
    rng = random.Random(12345)

    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_pts = TOTAL_POINTS

    # Precompute squared distances between all pairs
    # dist2[i][j] = squared distance between point i and point j
    dist2 = np.zeros((n_pts, n_pts), dtype=np.int32)
    for i in range(n_pts):
        xi, yi = _idx_to_point(i)
        for j in range(i + 1, n_pts):
            xj, yj = _idx_to_point(j)
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i][j] = d
            dist2[j][i] = d

    # Precompute: for each pair (i, j), the set of points k such that {i, j, k} is isosceles
    # Actually, we need: for a set S, a point q is blocked if there exist a, b in S such that {q, a, b} is isosceles.
    # 
    # Let's precompute: iso_mask[i][j] = bitmask of points k where {i, j, k} is isosceles
    # This is expensive: 256*256*256 = 16.7M checks. Let's be smarter.
    #
    # For points i, j, k: isosceles if d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(j,k)==d(i,k)
    # For fixed i, j: k is isosceles if d(i,k)==d(i,j) or d(j,k)==d(i,j) or d(i,k)==d(j,k)
    #
    # Let's precompute for each point i and each distance d, the set of points at distance d from i.
    # dist_sets[i][d] = bitmask of points at squared distance d from point i
    
    # Max squared distance in 16x16 grid: (15)^2 + (15)^2 = 450
    max_d = 2 * (N - 1) ** 2  # 450
    
    dist_sets = [dict() for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(n_pts):
            if i == j:
                continue
            d = dist2[i][j]
            if d not in dist_sets[i]:
                dist_sets[i][d] = 1 << j
            else:
                dist_sets[i][d] |= (1 << j)
    
    # Precompute iso_mask[i][j]: bitmask of k such that {i,j,k} is isosceles
    # {i,j,k} is isosceles if d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(j,k)==d(i,k)
    # For fixed i,j with d_ij = dist2[i][j]:
    #   k where d(i,k) == d_ij: dist_sets[i][d_ij]
    #   k where d(j,k) == d_ij: dist_sets[j][d_ij]
    #   k where d(i,k) == d(j,k): points equidistant from i and j (perpendicular bisector)
    #
    # The third condition is trickier. Let's just compute it directly for each (i,j) pair.
    # Actually, let's precompute the perpendicular bisector set.
    
    # For points i=(xi,yi) and j=(xj,yj), the perpendicular bisector consists of points k=(xk,yk)
    # such that (xk-xi)^2+(yk-yi)^2 = (xk-xj)^2+(yk-yj)^2
    # Expanding: -2*xk*xi - 2*yk*yi + xi^2+yi^2 = -2*xk*xj - 2*yk*yj + xj^2+yj^2
    # 2*xk*(xj-xi) + 2*yk*(yj-yi) = xj^2+yj^2 - xi^2-yi^2
    
    # Precompute perp_bis_mask[i][j]: bitmask of k equidistant from i and j
    perp_bis_mask = np.zeros((n_pts, n_pts), dtype=np.uint64)
    for i in range(n_pts):
        xi, yi = _idx_to_point(i)
        for j in range(n_pts):
            if i == j:
                perp_bis_mask[i][j] = (1 << n_pts) - 1  # all points equidistant from i to itself
                continue
            xj, yj = _idx_to_point(j)
            # 2*xk*(xj-xi) + 2*yk*(yj-yi) = xj^2+yj^2 - xi^2-yi^2
            ax = 2 * (xj - xi)
            ay = 2 * (yj - yi)
            rhs = xj*xj + yj*yj - xi*xi - yi*yi
            
            mask = 0
            for k in range(n_pts):
                xk, yk = _idx_to_point(k)
                if ax * xk + ay * yk == rhs:
                    mask |= (1 << k)
            perp_bis_mask[i][j] = mask
    
    # Now iso_mask[i][j] = dist_sets[i][d_ij] | dist_sets[j][d_ij] | perp_bis_mask[i][j]
    # But we need to exclude i and j themselves from consideration (they're already in the set)
    # Actually for blocking: q is blocked by {a,b} if {q,a,b} is isosceles, i.e., q is in iso_mask[a][b]
    
    # Let's precompute iso_block_mask[a][b] = set of points q (excluding a and b) such that {q,a,b} is isosceles
    # This is what we need for the hitting set formulation.
    
    # Actually, let's think about this differently.
    # A set S is maximal no-isosceles if:
    # 1. No three points in S form an isosceles triangle.
    # 2. For every point q not in S, there exist a, b in S such that {q, a, b} is isosceles.
    #
    # Condition 2 means: for every q not in S, q must be in the union of iso_block_mask[a][b] for all pairs (a,b) in S.
    #
    # So if we define for each point q, the set of pairs (a,b) that block q, then S must hit all these sets.
    # This is a hitting set / set cover problem.
    
    # Let's precompute: for each point q, which pairs (a,b) block it.
    # blocked_by[q] = set of (a,b) pairs such that {q,a,b} is isosceles
    
    # For efficiency, let's use a different approach:
    # For a given set S, a point q is blocked if there exist a,b in S with {q,a,b} isosceles.
    # We can check this by: for each a in S, check if there exists b in S such that {q,a,b} is isosceles.
    # {q,a,b} is isosceles iff d(q,a)==d(q,b) or d(q,a)==d(a,b) or d(q,b)==d(a,b).
    
    # Let's precompute for each pair (a,b): the set of q that are blocked by (a,b)
    # block_mask[a][b] = bitmask of q such that {q,a,b} is isosceles (q != a, q != b)
    
    block_mask = np.zeros((n_pts, n_pts), dtype=np.uint64)
    for a in range(n_pts):
        xa, ya = _idx_to_point(a)
        for b in range(a + 1, n_pts):
            xb, yb = _idx_to_point(b)
            d_ab = dist2[a][b]
            
            mask = 0
            for q in range(n_pts):
                if q == a or q == b:
                    continue
                xq, yq = _idx_to_point(q)
                d_aq = (xq - xa)**2 + (yq - ya)**2
                d_bq = (xq - xb)**2 + (yq - yb)**2
                if d_aq == d_bq or d_aq == d_ab or d_bq == d_ab:
                    mask |= (1 << q)
            block_mask[a][b] = mask
            block_mask[b][a] = mask
    
    # Now: a set S is maximal iff for every q not in S, there exist a,b in S such that q is in block_mask[a][b].
    # Equivalently: the union of block_mask[a][b] for all (a,b) in S must cover all points not in S.
    
    # For the no-isosceles condition: no three points in S form an isosceles triangle.
    # For a candidate point p to add to S: for all a,b in S, p must NOT be in block_mask[a][b]... 
    # Wait, that's not quite right. p can be added if for all pairs (a,b) in S, {p,a,b} is NOT isosceles.
    # {p,a,b} is isosceles iff p is in block_mask[a][b].
    # So p can be added iff p is not in block_mask[a][b] for any (a,b) in S.
    
    # Let's define: conflict_mask[p] = union of block_mask[a][b] over all (a,b) in S
    # Then p can be added iff p is not in conflict_mask... no wait.
    # p can be added iff for all (a,b) in S, p is NOT in block_mask[a][b].
    # i.e., p is not in the union of block_mask[a][b] for all (a,b) in S.
    
    # Let's define for a set S:
    # conflict[p] = OR of block_mask[a][b] for all (a,b) in S
    # Then p can be added to S iff (conflict >> p) & 1 == 0
    
    # And S is maximal iff for all q not in S, (conflict >> q) & 1 == 1
    
    # This is clean! Let's use this.
    
    def _can_add(p_idx, S_indices, conflict_cache):
        """Check if point p_idx can be added to set S_indices."""
        # p_idx can be added iff for all (a,b) in S, p_idx is not in block_mask[a][b]
        # i.e., p_idx is not in the conflict mask of S
        # conflict_cache is the OR of block_mask[a][b] for all (a,b) in S
        if (conflict_cache >> p_idx) & 1:
            return False
        return True
    
    def _compute_conflict(S_indices):
        """Compute the conflict mask for a set S_indices."""
        conflict = 0
        n = len(S_indices)
        for i in range(n):
            a = S_indices[i]
            for j in range(i + 1, n):
                b = S_indices[j]
                conflict |= block_mask[a][b]
        return conflict
    
    def _is_maximal(S_indices):
        """Check if S is a maximal no-isosceles set."""
        if len(S_indices) < 2:
            return False
        
        S_set = set(S_indices)
        conflict = _compute_conflict(S_indices)
        
        # Check no-isosceles: for all (a,b) in S, no c in S is in block_mask[a][b]
        # Actually, we need to check that no three points in S form an isosceles triangle.
        # This is equivalent to: for all (a,b) in S, no c in S (c != a, c != b) is in block_mask[a][b].
        for i in range(len(S_indices)):
            a = S_indices[i]
            for j in range(i + 1, len(S_indices)):
                b = S_indices[j]
                # Check if any other point in S is in block_mask[a][b]
                bm = block_mask[a][b]
                for k in S_indices:
                    if k != a and k != b and (bm >> k) & 1:
                        return False
        
        # Check maximality: for every point not in S, it must be blocked
        for q in range(n_pts):
            if q in S_set:
                continue
            if not ((conflict >> q) & 1):
                return False
        
        return True
    
    def _reduce_to_minimal(S_indices):
        """Try to remove points while maintaining maximality."""
        S_list = list(S_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(S_list)):
                p = S_list[i]
                S_test = S_list[:i] + S_list[i+1:]
                if len(S_test) < 2:
                    continue
                if _is_maximal(S_test):
                    S_list = S_test
                    changed = True
                    break
        return S_list
    
    # Greedy construction using hitting set / coverage approach
    def _greedy_construct(rng, strategy='random'):
        """
        Greedily construct a maximal no-isosceles set.
        strategy: 'random' - random order, 'coverage' - pick point that blocks most
        """
        chosen = []
        chosen_set = set()
        conflict = 0  # OR of block_mask[a][b] for all pairs in chosen
        
        if strategy == 'random':
            order = list(range(n_pts))
            rng.shuffle(order)
            for p in order:
                if _can_add(p, chosen, conflict):
                    # Update conflict
                    for a in chosen:
                        conflict |= block_mask[p][a]
                    chosen.append(p)
                    chosen_set.add(p)
            # Now ensure maximality: add points until all are blocked
            changed = True
            while changed:
                changed = False
                for q in range(n_pts):
                    if q in chosen_set:
                        continue
                    if not ((conflict >> q) & 1):
                        # q is not blocked, add it
                        if _can_add(q, chosen, conflict):
                            for a in chosen:
                                conflict |= block_mask[q][a]
                            chosen.append(q)
                            chosen_set.add(q)
                            changed = True
        else:  # coverage
            remaining = set(range(n_pts))
            while remaining:
                best_p = None
                best_score = -1
                for p in remaining:
                    if not _can_add(p, chosen, conflict):
                        continue
                    # Score: how many new points would be blocked
                    new_conflict = conflict
                    for a in chosen:
                        new_conflict |= block_mask[p][a]
                    new_blocked = bin(new_conflict & ~conflict).count('1')
                    if new_blocked > best_score:
                        best_score = new_blocked
                        best_p = p
                if best_p is None:
                    break
                for a in chosen:
                    conflict |= block_mask[best_p][a]
                chosen.append(best_p)
                chosen_set.add(best_p)
                remaining.remove(best_p)
            # Ensure maximality
            changed = True
            while changed:
                changed = False
                for q in range(n_pts):
                    if q in chosen_set:
                        continue
                    if not ((conflict >> q) & 1):
                        if _can_add(q, chosen, conflict):
                            for a in chosen:
                                conflict |= block_mask[q][a]
                            chosen.append(q)
                            chosen_set.add(q)
                            changed = True
        
        return chosen
    
    # Local search: try to swap one point for another to reduce size
    def _local_search(S_indices, time_limit=60.0):
        start_time = time.time()
        S_list = list(S_indices)
        n_S = len(S_list)
        
        for iteration in range(20):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            for i in range(len(S_list)):
                if time.time() - start_time > time_limit:
                    break
                p_remove = S_list[i]
                S_without = S_list[:i] + S_list[i+1:]
                if len(S_without) < 2:
                    continue
                
                # Compute conflict without p_remove
                conflict_without = _compute_conflict(S_without)
                S_without_set = set(S_without)
                
                # Try adding each point not in S_without
                for p_add in range(n_pts):
                    if p_add in S_without_set:
                        continue
                    if p_add == p_remove:
                        continue
                    
                    if not _can_add(p_add, S_without, conflict_without):
                        continue
                    
                    S_new = S_without + [p_add]
                    if _is_maximal(S_new):
                        reduced = _reduce_to_minimal(S_new)
                        if len(reduced) < len(S_list):
                            S_list = reduced
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return S_list
    
    # Main search
    best_set = None
    best_size = n_pts + 1
    
    # Strategy 1: Multiple random greedy attempts
    for attempt in range(300):
        if time.time() - start_time > DEADLINE:
            break
        
        chosen = _greedy_construct(rng, 'random')
        if not chosen:
            continue
        
        reduced = _reduce_to_minimal(chosen)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 6:
            break
    
    # Strategy 2: Coverage greedy
    for attempt in range(100):
        if time.time() - start_time > DEADLINE:
            break
        
        chosen = _greedy_construct(rng, 'coverage')
        if not chosen:
            continue
        
        reduced = _reduce_to_minimal(chosen)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 6:
            break
    
    # Strategy 3: Local search on best found
    if best_set is not None and time.time() - start_time < DEADLINE - 30:
        improved = _local_search(best_set, time_limit=DEADLINE - time.time() - 10)
        if improved is not None and len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        for p in range(n_pts):
            if all(not ((block_mask[a][b] >> p) & 1) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        best_set = chosen
        best_size = len(chosen)
    
    # Convert back to (x, y) tuples
    return [_idx_to_point(idx) for idx in best_set]


if __name__ == '__main__':
    result = construct()
    print(f"Found set of size {len(result)}")
