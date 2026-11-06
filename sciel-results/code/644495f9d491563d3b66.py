"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _line_key_from_two(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2 where ax+by=c."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

class PointSet:
    """Maintains 16 points with incremental line tracking for fast scoring."""
    
    def __init__(self, points):
        self.points = [p[:] for p in points]
        self.n = 16
        self._rebuild()
    
    def _rebuild(self):
        """Rebuild all line mappings from scratch."""
        self.line_pts = defaultdict(set)
        self.exact4_count = 0
        self.has5 = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                key = _line_key_from_two(self.points[i], self.points[j])
                if key is None:
                    continue
                self.line_pts[key].add(i)
                self.line_pts[key].add(j)
        self._count_lines()
    
    def _count_lines(self):
        """Count exact-4 lines and check for 5+ collinear."""
        self.exact4_count = 0
        self.has5 = False
        for key, pts in self.line_pts.items():
            cnt = len(pts)
            if cnt == 4:
                self.exact4_count += 1
            elif cnt >= 5:
                self.has5 = True
    
    def move_point(self, idx, new_pt):
        """Move point idx to new_pt, updating line mappings incrementally."""
        old_pt = self.points[idx][:]
        
        # Remove idx from all lines that contained it
        # First, find all lines involving idx
        lines_to_remove = []
        for key in list(self.line_pts.keys()):
            if idx in self.line_pts[key]:
                lines_to_remove.append(key)
        
        for key in lines_to_remove:
            self.line_pts[key].discard(idx)
            if not self.line_pts[key]:
                del self.line_pts[key]
        
        # Place new point
        self.points[idx] = new_pt[:]
        
        # Add idx to all lines through new point with other points
        for j in range(self.n):
            if j == idx:
                continue
            key = _line_key_from_two(new_pt, self.points[j])
            if key is None:
                continue
            self.line_pts[key].add(idx)
            self.line_pts[key].add(j)
        
        # Recount
        self._count_lines()
    
    def score(self):
        return self.exact4_count, self.has5
    
    def get_lines_with_count(self, target_count):
        """Return list of (key, set_of_indices) for lines with exactly target_count points."""
        result = []
        for key, pts in self.line_pts.items():
            if len(pts) == target_count:
                result.append((key, pts))
        return result
    
    def get_lines_with_at_least(self, min_count):
        """Return list of (key, set_of_indices) for lines with at least min_count points."""
        result = []
        for key, pts in self.line_pts.items():
            if len(pts) >= min_count:
                result.append((key, pts))
        return result

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    ps = PointSet(points)
    return ps.score()

def _score_points_fast(points):
    """Faster scoring using PointSet."""
    ps = PointSet(points)
    return ps.score()

def _find_good_positions(ps, idx, rng, max_candidates=20):
    """Find candidate positions for point idx that might improve the score."""
    candidates = []
    old_pt = ps.points[idx]
    
    # Strategy 1: Lines with exactly 3 other points - extending to make 4
    for key, pts in ps.line_pts.items():
        if idx in pts:
            continue
        if len(pts) == 3:
            # Find the line equation
            a, b, c = key
            # Try to find integer points on this line
            # Get the 3 points to determine the line's extent
            point_indices = list(pts)
            p1 = ps.points[point_indices[0]]
            p2 = ps.points[point_indices[1]]
            
            # Direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            
            # Try positions along the line near existing points
            for pi in point_indices:
                px, py = ps.points[pi]
                for step in range(-3, 4):
                    if step == 0:
                        continue
                    nx = px + step * dx_n
                    ny = py + step * dy_n
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        new_pt = [nx, ny]
                        if new_pt != old_pt:
                            if not any(new_pt == ps.points[i] for i in range(ps.n)):
                                # Verify it's on the line
                                if a * nx + b * ny == c:
                                    candidates.append(new_pt)
    
    # Strategy 2: Random nearby positions
    for _ in range(10):
        nx = old_pt[0] + rng.randint(-8, 8)
        ny = old_pt[1] + rng.randint(-8, 8)
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            new_pt = [nx, ny]
            if new_pt != old_pt:
                if not any(new_pt == ps.points[i] for i in range(ps.n)):
                    candidates.append(new_pt)
    
    # Strategy 3: Try to create a new 4-line by placing on a line with 3 points
    # (already covered in Strategy 1)
    
    if not candidates:
        return None
    
    return rng.choice(candidates[:max_candidates])

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    best_ps = PointSet(base)
    best_score, _ = best_ps.score()
    best_points = [p[:] for p in best_ps.points]
    
    rng = random.Random(99)
    
    # Multiple restarts with different strategies
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize current configuration
        if restart == 0:
            current_ps = PointSet(base)
        elif restart == 1:
            # Perturbed grid
            pts = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(rng.randint(3, 8)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-3, 3), pts[idx][1] + rng.randint(-3, 3)]
            # Ensure distinctness
            unique = []
            seen = set()
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            while len(unique) < 16:
                x = rng.randint(-5, 5)
                y = rng.randint(-5, 5)
                if [x, y] not in seen:
                    seen.add([x, y])
                    unique.append([x, y])
            current_ps = PointSet(unique[:16])
        elif restart % 3 == 0:
            # From best
            current_ps = PointSet(best_points)
        else:
            # Random perturbation of best
            pts = [p[:] for p in best_points]
            for _ in range(rng.randint(2, 6)):
                idx = rng.randint(0, 15)
                pts[idx] = [pts[idx][0] + rng.randint(-4, 4), pts[idx][1] + rng.randint(-4, 4)]
            # Ensure distinctness
            unique = []
            seen = set()
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            while len(unique) < 16:
                x = rng.randint(-5, 5)
                y = rng.randint(-5, 5)
                if [x, y] not in seen:
                    seen.add([x, y])
                    unique.append([x, y])
            current_ps = PointSet(unique[:16])
        
        s, h5 = current_ps.score()
        if h5:
            continue
        current_score = s
        
        # Simulated annealing
        T = 0.8
        no_improve = 0
        max_no_improve = 500
        step = 0
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            step += 1
            
            idx = rng.randint(0, 15)
            old_pt = current_ps.points[idx][:]
            
            # Choose new position
            use_smart = rng.random() < 0.4
            if use_smart:
                new_pt = _find_good_positions(current_ps, idx, rng)
                if new_pt is None:
                    new_pt = [old_pt[0] + rng.randint(-6, 6), old_pt[1] + rng.randint(-6, 6)]
            else:
                new_pt = [old_pt[0] + rng.randint(-6, 6), old_pt[1] + rng.randint(-6, 6)]
            
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            if any(new_pt == current_ps.points[i] for i in range(16)):
                continue
            
            # Try the move
            current_ps.move_point(idx, new_pt)
            s_new, h5_new = current_ps.score()
            
            if h5_new:
                # Revert
                current_ps.move_point(idx, old_pt)
                no_improve += 1
                continue
            
            if s_new >= current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current_ps.points]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                # Revert
                current_ps.move_point(idx, old_pt)
                no_improve += 1
            
            T = max(0.05, T * 0.9985)
    
    return best_points
# EVOLVE-BLOCK-END
