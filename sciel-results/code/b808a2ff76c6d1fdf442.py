"""Direct geometric construction for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START


def _score(points):
    """Feasible common radius using KDTree for nearest-neighbor distance."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)


def _score_local(points, idx, old_pos, new_pos, tree, cached_dists):
    """
    Fast score after moving point idx from old_pos to new_pos.
    Uses cached KDTree distances for all other pairs; only recomputes
    distances involving idx. Returns the new feasible radius.
    """
    # New boundary for moved point
    new_norm = np.linalg.norm(new_pos)
    # The minimum boundary is min over all points of (1 - |c_i|)
    # For un-moved points, their (1-|c_i|) is unchanged; for moved point it's (1-new_norm)
    # We need the global min boundary. We can track: old_min_boundary was cached.
    # But simpler: compute new boundary for moved point, and the rest is unchanged.
    # Actually we need min over ALL i of (1 - |c_i|).
    # Let's just compute it properly but cheaply:
    # norms for all other points are unchanged; only idx changes.
    # We'll pass in the old min_boundary and old norm of idx.
    # For simplicity in this version, recompute norms (cheap for N=300).
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0

    # For pair distances: all pairs not involving idx are unchanged.
    # Pairs involving idx: distance from new_pos to points[j] for all j != idx.
    # We need min of: (cached min pair dist among non-idx pairs), and (min_j!=idx |new_pos - points[j]|)
    # cached_dists[:, 1] gives nearest neighbor distance for each point.
    # The global min pair distance is min(cached_dists[:, 1]).
    # But if the moved point was involved in the minimum pair, we need to update.
    # Simpler approach: compute all distances from new_pos to other points.
    diff = points - new_pos
    diff[idx, :] = 0.0  # exclude self
    dists_to_others = np.linalg.norm(diff, axis=1)
    dists_to_others[idx] = np.inf  # exclude self
    min_new_pair = float(np.min(dists_to_others))

    # For pairs not involving idx, the min is the minimum of cached_dists[:,1]
    # but we must exclude pairs where idx was the nearest neighbor of some point,
    # or where idx was the second nearest. This is complex.
    # For N=300, just compute the full min pair distance cheaply:
    # min_pair = min( min over j!=idx of |new_pos - points[j]|, min over pairs not involving idx )
    # The latter: from cached_dists, the min among all points' nearest-neighbor dist,
    # but we need to be careful if idx was the nearest neighbor of some point.
    # Given N=300, let's just do a vectorized min over all pairs not involving idx.
    # Actually, for N=300, a full pairwise is 300*299/2 = 44850 ops, still fast.
    # But let's try the cached approach: the min pair dist not involving idx is
    # the min of cached_dists[:,1] for points whose nearest neighbor is not idx,
    # plus for points whose nearest neighbor IS idx, their second-nearest distance.
    # This is getting complex. For N=300, let's just compute it directly but vectorized.

    # Full pairwise min excluding self, but only need min:
    # min over all i<j of |points[i]-points[j]|
    # Vectorized: compute all pairwise distances, take min.
    # For N=300 this is fine.
    # Actually, let's use the fact that only distances involving idx changed.
    # min_pair = min( min_j!=idx |new_pos - points[j]|, min_pair_not_involving_idx )
    # min_pair_not_involving_idx: from cached_dists, for each point j, if its nearest
    # neighbor is not idx, then cached_dists[j,1] is valid. If its nearest neighbor IS idx,
    # then we need its second nearest (which we don't have cached).
    # Simplest correct approach for N=300: just compute the full min.
    # It's 44850 distance computations, ~0.1ms.

    # Even simpler: since we're doing this in a tight loop, let's precompute
    # the min pair distance not involving idx once per outer iteration.
    # But for now, let's just do the full computation - it's still fast.

    # Compute min pair distance properly:
    # We have dists_to_others (dist from new_pos to all others).
    # For pairs not involving idx: use cached info.
    # Let's just compute min of:
    #   a) min_j!=idx |new_pos - points[j]|  = min_new_pair
    #   b) min over i<j, i,j != idx of |points[i]-points[j]|
    # For (b), we can use: min of cached_dists[:,1] where the nearest neighbor is not idx,
    # and for points whose nearest neighbor is idx, we need their second nearest.
    # To keep it simple and correct, let's just do the full pairwise min for (b).
    # With N=300, this is acceptable.

    # Actually, let me reconsider. The full _score with KDTree takes maybe 0.5ms.
    # The local score with full pairwise min takes similar time.
    # The real speedup comes from NOT rebuilding the KDTree.
    # Let me just compute the min pair distance via broadcasting for N=300.

    # Min pair distance not involving idx:
    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    sub_pts = points[mask]
    n_sub = len(sub_pts)
    if n_sub >= 2:
        # Compute min pairwise distance in sub_pts
        # Using KDTree on sub_pts would be fast
        sub_tree = cKDTree(sub_pts)
        sub_dists, _ = sub_tree.query(sub_pts, k=2)
        min_sub_pair = float(np.min(sub_dists[:, 1]))
    else:
        min_sub_pair = np.inf

    min_pair = min(min_new_pair, min_sub_pair)
    return min(min_boundary, min_pair / 2.0)


def _build_hex_rings(spacing, ring_radii=None):
    """
    Build N points in concentric hexagonal rings with given spacing.
    Each ring holds points at a given radius, placed to maximize min pair distance
    (uniform angles). The lattice spacing determines ring radii.
    """
    pts = []
    if ring_radii is None:
        # Determine ring radii from hex geometry: rings at sqrt(3)*spacing * k/2
        # Actually standard hex rings: distance from center = spacing * k for ring k
        # For a hex lattice, points in ring k are at distance k*spacing (for axial)
        # We'll just use evenly spaced radii tuned to spacing.
        n_rings = int(1.0 / spacing) + 1
        ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    # Distribute N points across rings: outer rings get more (circumference)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
    # Adjust to match N
    if total < N:
        deficit = N - total
        # Add to outer rings
        for i in range(len(counts) - 1, -1, -1):
            add = min(deficit, max(1, counts[i] // 2))
            counts[i] += add
            deficit -= add
            if deficit <= 0:
                break
    elif total > N:
        excess = total - N
        for i in range(len(counts) - 1, -1, -1):
            rem = min(excess, counts[i] - (1 if i == 0 and ring_radii[i] < 0.1 else 6))
            counts[i] -= rem
            excess -= rem
            if excess <= 0:
                break
    # Build points
    for k, r in enumerate(ring_radii):
        c = counts[k]
        if c <= 0:
            continue
        if c == 1 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c + (np.pi / c) * (k % 2)
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    pts = np.vstack(pts)
    if len(pts) != N:
        # Fallback fill
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


def _build_hex_lattice(spacing):
    """Hex lattice points inside unit disk, pick N closest to center."""
    k = int(1.0 / spacing) + 4
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.3:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _refine(points, deadline, max_iters=8000, base_step=0.012):
    """Local refinement with cheap local feasibility check before full score."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(7)
    step = base_step
    # Precompute neighbor info for cheap local checks
    tree = cKDTree(points)
    # For each point, get its 5 nearest neighbors and distances
    nn_dists, nn_idx = tree.query(points, k=6)  # k=6 includes self
    # nn_dists[:,0] = 0 (self), nn_dists[:,1:] = distances to 5 nearest
    nn_idx = nn_idx[:, 1:]  # indices of 5 nearest neighbors
    nn_dists = nn_dists[:, 1:]  # distances to 5 nearest neighbors

    for it in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r

        # Try several candidate moves
        n_tries = 12
        for t in range(n_tries):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.3, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])

            # Cheap local check: verify cand doesn't violate constraints
            # with its known neighbors and the boundary
            cand_norm = np.linalg.norm(cand)
            if cand_norm > 1.0:
                continue  # violates boundary

            # Check against 5 nearest neighbors (and a few random others for safety)
            local_ok = True
            for j in nn_idx[idx]:
                d = np.linalg.norm(cand - points[j])
                if d < 2.0 * best_r * 0.99:  # must maintain current radius with margin
                    local_ok = False
                    break
            if not local_ok:
                continue

            # Also check a few random points to catch non-local violations
            for _ in range(3):
                j = rng.integers(0, N)
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < 2.0 * best_r * 0.99:
                    local_ok = False
                    break
            if not local_ok:
                continue

            # Full score check
            points[idx] = cand
            r = _score(points)
            if r > best_local_r:
                best_local_r = r
                best_local = cand.copy()
                # Update neighbor info for this point
                tree = cKDTree(points)
                nn_dists, nn_idx_tmp = tree.query(points, k=6)
                nn_idx = nn_idx_tmp[:, 1:]
                nn_dists = nn_dists[:, 1:]
            else:
                points[idx] = original

        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            step = max(0.003, step * 0.997)  # slow decay
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    candidates = []

    # Hex lattice over a range of spacings - finer grid around optimal
    for trial in range(40):
        spacing = 0.052 + trial * 0.003
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # Hex rings with tuned radii
    for trial in range(25):
        spacing = 0.055 + trial * 0.003
        pts = _build_hex_rings(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Refine top candidates with more budget
    n_refine = min(5, len(candidates))
    for cand_idx in range(n_refine):
        if time.time() > deadline - 10.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        # Give more iterations to better candidates
        max_it = 6000 if cand_idx == 0 else 4000
        refined_pts, refined_r = _refine(cand_pts, deadline, max_iters=max_it,
                                         base_step=0.012)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Final polish: one more refinement pass on the best result with smaller steps
    if time.time() < deadline - 5.0:
        final_pts, final_r = _refine(best_points, deadline, max_iters=3000,
                                     base_step=0.006)
        if final_pts is not None and final_r > best_r:
            best_points, best_r = final_pts, final_r

    return best_points


# EVOLVE-BLOCK-END
