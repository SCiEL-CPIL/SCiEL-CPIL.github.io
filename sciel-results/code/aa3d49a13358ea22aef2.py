"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter

def _build_triple_indices(n):
    """Precompute all C(n,3) triple indices as three arrays."""
    i_list, j_list, k_list = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_list.append(i)
                j_list.append(j)
                k_list.append(k)
    return (np.array(i_list, dtype=np.int32),
            np.array(j_list, dtype=np.int32),
            np.array(k_list, dtype=np.int32))

def _compute_areas(pts_arr, tri_i, tri_j, tri_k):
    """Compute doubled-area (absolute determinant) for all triples."""
    x = pts_arr[:, 0]
    y = pts_arr[:, 1]
    xi, yi = x[tri_i], y[tri_i]
    xj, yj = x[tri_j], y[tri_j]
    xk, yk = x[tri_k], y[tri_k]
    dets = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return dets

def _full_evaluate(points):
    """Full evaluation: return (score, histogram_dict, points_array)."""
    pts = np.array(points, dtype=np.int32)
    n = len(pts)
    tri_i, tri_j, tri_k = _build_triple_indices(n)
    dets = _compute_areas(pts, tri_i, tri_j, tri_k)
    # Only count positive areas
    pos = dets[dets > 0]
    if len(pos) == 0:
        return 0, {}, pts
    counts = Counter(pos.tolist())
    score = max(counts.values())
    return score, dict(counts), pts

class _FastEvaluator:
    """
    Incremental evaluator: precomputes triple indices and maintains
    a histogram of doubled-area values. Moving one point only requires
    recomputing ~435 triples.
    """
    def __init__(self, points):
        self.n = len(points)
        self.pts = np.array(points, dtype=np.int32)
        self.tri_i, self.tri_j, self.tri_k = _build_triple_indices(self.n)
        self._compute_all()
        self._build_point_masks()
    
    def _compute_all(self):
        dets = _compute_areas(self.pts, self.tri_i, self.tri_j, self.tri_k)
        self.dets = dets
        self._update_histogram()
    
    def _build_point_masks(self):
        """For each point index, precompute which triples include it."""
        n = self.n
        m = len(self.tri_i)
        self.point_masks = []
        for idx in range(n):
            mask = (self.tri_i == idx) | (self.tri_j == idx) | (self.tri_k == idx)
            self.point_masks.append(mask)
    
    def _update_histogram(self):
        pos = self.dets[self.dets > 0]
        if len(pos) == 0:
            self.hist = {}
            self.score = 0
            return
        self.hist = dict(Counter(pos.tolist()))
        self.score = max(self.hist.values())
    
    def _recompute_point(self, idx):
        """Recompute areas for all triples involving point idx."""
        mask = self.point_masks[idx]
        ti, tj, tk = self.tri_i[mask], self.tri_j[mask], self.tri_k[mask]
        x = self.pts[:, 0]
        y = self.pts[:, 1]
        xi, yi = x[ti], y[ti]
        xj, yj = x[tj], y[tj]
        xk, yk = x[tk], y[tk]
        new_dets = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        self.dets[mask] = new_dets
    
    def _update_histogram_fast(self):
        """Update histogram incrementally is complex; just rebuild from dets."""
        pos = self.dets[self.dets > 0]
        if len(pos) == 0:
            self.hist = {}
            self.score = 0
            return
        # Use numpy for speed
        vals, counts = np.unique(pos, return_counts=True)
        self.hist = {int(v): int(c) for v, c in zip(vals, counts)}
        self.score = int(counts.max())
    
    def try_move(self, idx, new_p):
        """
        Try moving point idx to new_p. Returns (improved, new_score).
        Modifies self.pts if improved.
        """
        new_p = np.array(new_p, dtype=np.int32)
        old_p = self.pts[idx].copy()
        
        # Check bounds
        if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
            return False, self.score
        
        # Check collision
        for i in range(self.n):
            if i != idx and self.pts[i][0] == new_p[0] and self.pts[i][1] == new_p[1]:
                return False, self.score
        
        # Apply move
        self.pts[idx] = new_p
        self._recompute_point(idx)
        self._update_histogram_fast()
        new_score = self.score
        
        if new_score > self.score:
            return True, new_score
        else:
            # Revert
            self.pts[idx] = old_p
            self._recompute_point(idx)
            self._update_histogram_fast()
            return False, self.score
    
    def get_points(self):
        return [tuple(p) for p in self.pts]


def _local_search_fast(points, time_limit=30.0, radius=4, seed=42):
    """
    Fast local search using incremental evaluation.
    """
    start_time = time.time()
    rng = random.Random(seed)
    n = len(points)
    
    eval_obj = _FastEvaluator(points)
    best_score = eval_obj.score
    best_points = eval_obj.get_points()
    
    # Track recent moves for diversity
    move_radius = radius
    
    while time.time() - start_time < time_limit:
        # Adaptive radius: increase if stuck
        idx = rng.randint(0, n - 1)
        ox, oy = points[idx]
        
        # Random offset
        dx = rng.randint(-move_radius, move_radius)
        dy = rng.randint(-move_radius, move_radius)
        if dx == 0 and dy == 0:
            continue
        new_p = (ox + dx, oy + dy)
        
        improved, new_score = eval_obj.try_move(idx, new_p)
        if improved:
            best_score = new_score
            best_points = eval_obj.get_points()
            # Slightly expand radius after success
            if move_radius < 8:
                move_radius += 1
        else:
            # Occasionally shrink radius to explore more finely
            if move_radius > 2 and rng.random() < 0.1:
                move_radius -= 1
    
    return best_points, best_score


def _multi_restart(points, time_limit=30.0, seed=123):
    """
    Multi-restart local search with random perturbations.
    """
    start_time = time.time()
    rng = random.Random(seed)
    n = len(points)
    best_points = list(points)
    best_score = _full_evaluate(best_points)[0]
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        # Create perturbation
        perturbed = list(best_points)
        num_moves = rng.randint(2, 5)
        for _ in range(num_moves):
            idx = rng.randint(0, n - 1)
            ox, oy = perturbed[idx]
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
            new_p = (ox + dx, oy + dy)
            if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30:
                conflict = False
                for i, p in enumerate(perturbed):
                    if i != idx and p == new_p:
                        conflict = True
                        break
                if not conflict:
                    perturbed[idx] = new_p
        
        # Time budget per restart
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1.0:
            break
        per_restart = min(3.0, remaining / max(1, 3 - num_restarts % 3))
        
        refined, score = _local_search_fast(perturbed, time_limit=per_restart, seed=seed + num_restarts)
        if score > best_score:
            best_score = score
            best_points = refined
        num_restarts += 1
    
    return best_points, best_score


def _generate_grid_candidates():
    """Generate diverse grid-based initial configurations."""
    candidates = []
    
    # Standard grids
    for rows in range(2, 9):
        for cols in range(2, 16):
            if rows * cols < 30:
                continue
            pts = [(x, y) for x in range(cols) for y in range(rows)]
            if len(pts) > 30:
                pts = pts[:30]
            elif len(pts) < 30:
                continue
            if len(pts) == 30:
                candidates.append(pts)
    
    # Grids with shifted rows (staggered)
    for rows in range(2, 8):
        for cols in range(4, 12):
            pts = []
            for y in range(rows):
                offset = (y % 2) * (cols // 2)
                for x in range(cols):
                    pts.append((x + offset, y))
            if len(pts) > 30:
                pts = pts[:30]
            if len(pts) == 30:
                candidates.append(pts)
    
    # Concentric rings / structured patterns
    # 5x6 with corners removed and replaced
    base = [(x, y) for x in range(6) for y in range(5)]
    variants = [
        [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),
         (0,1),(1,1),(2,1),(3,1),(4,1),(5,1),
         (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),
         (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),
         (0,4),(1,4),(2,4),(3,4),(4,4),(5,4)]
    ]
    candidates.extend(variants)
    
    # Remove duplicates
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Phase 1: Evaluate diverse initial candidates
    candidates = _generate_grid_candidates()
    
    best_points = None
    best_score = 0
    
    # Evaluate a subset of candidates (limited by time)
    max_eval = 200
    for i, pts in enumerate(candidates[:max_eval]):
        if time.time() > deadline - 20.0:
            break
        if len(pts) != 30:
            continue
        valid = all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts)
        if not valid:
            continue
        score, _, _ = _full_evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _full_evaluate(best_points)[0]
    
    # Phase 2: Fast local search from best candidate
    if time.time() < deadline - 10.0:
        time_remaining = deadline - time.time() - 5.0
        refined, score = _local_search_fast(best_points, time_limit=time_remaining, radius=4, seed=42)
        if score > best_score:
            best_score = score
            best_points = refined
    
    # Phase 3: Multi-restart for further improvement
    if time.time() < deadline - 3.0:
        time_remaining = deadline - time.time() - 2.0
        multi_pts, multi_score = _multi_restart(best_points, time_limit=time_remaining, seed=123)
        if multi_score > best_score:
            best_score = multi_score
            best_points = multi_pts
    
    # Phase 4: Final polish with tighter neighborhood
    if time.time() < deadline - 1.0:
        time_remaining = deadline - time.time() - 0.5
        if time_remaining > 1.0:
            polished, pol_score = _local_search_fast(best_points, time_limit=time_remaining, radius=2, seed=999)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished
    
    return best_points
# EVOLVE-BLOCK-END
