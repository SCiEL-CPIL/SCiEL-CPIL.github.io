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
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
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

class PointSetManager:
    """Manages a set of points and their line incidences for fast scoring and candidate generation."""
    
    def __init__(self):
        self.points = []
        self.point_set = set()
        self.line_pts = defaultdict(set) # key: (a,b,c), value: set of point indices
        self.exact4_count = 0
        self.has5 = False
        self._recount()

    def _recount(self):
        self.exact4_count = 0
        self.has5 = False
        for pts in self.line_pts.values():
            cnt = len(pts)
            if cnt == 4:
                self.exact4_count += 1
            elif cnt >= 5:
                self.has5 = True

    def add_point(self, pt):
        """Add a point, update structures. Returns (new_exact4, has5)."""
        if pt in self.point_set:
            return self.exact4_count, self.has5
        
        idx = len(self.points)
        self.points.append(pt)
        self.point_set.add(pt)
        
        # Update lines
        for j in range(idx):
            key = _line_key_from_two(pt, self.points[j])
            if key:
                self.line_pts[key].add(idx)
                self.line_pts[key].add(j)
        
        self._recount()
        return self.exact4_count, self.has5

    def remove_point(self, idx):
        """Remove point at index idx."""
        pt = self.points[idx]
        self.point_set.discard(pt)
        
        # Remove from lines
        lines_to_clean = []
        for key, pts in self.line_pts.items():
            if idx in pts:
                lines_to_clean.append(key)
        
        for key in lines_to_clean:
            self.line_pts[key].discard(idx)
            if not self.line_pts[key]:
                del self.line_pts[key]
        
        self.points.pop(idx)
        # Note: This shifts indices for all subsequent points. 
        # For a greedy builder, we only add, so we don't need complex removal/re-indexing 
        # unless we do backtracking. For simplicity in the greedy phase, we only add.
        # For local search, we will use a different mechanism or rebuild.
        
        self._recount()

    def get_candidates(self, rng, max_candidates=50):
        """Generate candidate points for the next addition."""
        candidates = set()
        
        # 1. Extend lines defined by pairs of existing points
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.points[i]
                p2 = self.points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n, dy_n = dx // g, dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n, dy_n = -dx_n, -dy_n
                
                # Try a few steps in both directions from existing points
                for base_pt in [p1, p2]:
                    for step in range(1, 4):
                        for sign in [-1, 1]:
                            nx = base_pt[0] + sign * step * dx_n
                            ny = base_pt[1] + sign * step * dy_n
                            if abs(nx) <= 1000 and abs(ny) <= 1000:
                                pt = (nx, ny)
                                if pt not in self.point_set:
                                    candidates.add(pt)
        
        # 2. Complete 3-point lines to 4
        # Find lines with exactly 3 points
        for key, pts in self.line_pts.items():
            if len(pts) == 3:
                a, b, c = key
                # Get the 3 points
                p_indices = list(pts)
                p1 = self.points[p_indices[0]]
                p2 = self.points[p_indices[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx_n, dy_n = dx // g, dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n, dy_n = -dx_n, -dy_n
                
                # Find integer points on this line that are not in the set
                # We can just scan a small range around the points
                min_x = min(p[0] for p in self.points) - 10
                max_x = max(p[0] for p in self.points) + 10
                min_y = min(p[1] for p in self.points) - 10
                max_y = max(p[1] for p in self.points) + 10
                
                # Efficiently find points on line ax+by=c in range
                # If a != 0, x = (c - by) / a
                # If b != 0, y = (c - ax) / b
                # We'll just iterate over a small window around the existing points
                for pi in p_indices:
                    px, py = self.points[pi]
                    for step in range(-5, 6):
                        if step == 0: continue
                        nx = px + step * dx_n
                        ny = py + step * dy_n
                        if abs(nx) <= 1000 and abs(ny) <= 1000:
                            pt = (nx, ny)
                            if pt not in self.point_set:
                                # Verify it's on the line
                                if a * nx + b * ny == c:
                                    candidates.add(pt)

        # 3. Random candidates
        for _ in range(20):
            nx = rng.randint(-100, 100)
            ny = rng.randint(-100, 100)
            pt = (nx, ny)
            if pt not in self.point_set:
                candidates.add(pt)

        return list(candidates)[:max_candidates]

def _local_search(ps_points, rng, deadline, current_best_score, current_best_points):
    """Perform local search on a given set of 16 points."""
    # Convert to the PointSet structure used in original code for consistency, 
    # but we can also just use a simple rebuild approach for scoring in the loop
    # To be fast, we use the incremental PointSet from the original code (fixed)
    
    # We need a class that supports move_point. 
    # Let's reuse the logic from the failed code but fixed.
    
    class LocalSearchPointSet:
        def __init__(self, points):
            self.points = [p[:] for p in points]
            self.n = 16
            self.line_pts = defaultdict(set)
            self.exact4_count = 0
            self.has5 = False
            self._rebuild()
            
        def _rebuild(self):
            self.line_pts = defaultdict(set)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    key = _line_key_from_two(self.points[i], self.points[j])
                    if key:
                        self.line_pts[key].add(i)
                        self.line_pts[key].add(j)
            self._count_lines()
            
        def _count_lines(self):
            self.exact4_count = 0
            self.has5 = False
            for pts in self.line_pts.values():
                cnt = len(pts)
                if cnt == 4:
                    self.exact4_count += 1
                elif cnt >= 5:
                    self.has5 = True
                    
        def move_point(self, idx, new_pt):
            old_pt = self.points[idx][:]
            
            # Remove idx from lines
            lines_to_remove = []
            for key in list(self.line_pts.keys()):
                if idx in self.line_pts[key]:
                    lines_to_remove.append(key)
            for key in lines_to_remove:
                self.line_pts[key].discard(idx)
                if not self.line_pts[key]:
                    del self.line_pts[key]
            
            self.points[idx] = new_pt[:]
            
            # Add idx to new lines
            for j in range(self.n):
                if j == idx: continue
                key = _line_key_from_two(new_pt, self.points[j])
                if key:
                    self.line_pts[key].add(idx)
                    self.line_pts[key].add(j)
                    
            self._count_lines()
            
        def score(self):
            return self.exact4_count, self.has5

    ps = LocalSearchPointSet(ps_points)
    s, h5 = ps.score()
    if h5:
        return s, ps.points # Invalid, but return anyway
    
    current_score = s
    T = 0.8
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        idx = rng.randint(0, 15)
        old_pt = ps.points[idx][:]
        
        # Smart move: try to complete a 3-line or extend a 2-line
        # For simplicity, random move with small step
        new_pt = [old_pt[0] + rng.randint(-5, 5), old_pt[1] + rng.randint(-5, 5)]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            continue
        if new_pt in [p for i, p in enumerate(ps.points) if i != idx]: # Check distinctness
            continue
            
        ps.move_point(idx, new_pt)
        s_new, h5_new = ps.score()
        
        if h5_new:
            ps.move_point(idx, old_pt)
            no_improve += 1
            continue
        
        if s_new >= current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
            if s_new > current_score:
                no_improve = 0
                current_score = s_new
            else:
                no_improve += 1
        else:
            ps.move_point(idx, old_pt)
            no_improve += 1
            
        T = max(0.05, T * 0.998)
        
    return current_score, ps.points

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    best_score = 0
    best_points = None
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Phase 1: Greedy Construction
        manager = PointSetManager()
        
        # Add first 2 points randomly to start lines
        p1 = (rng.randint(-50, 50), rng.randint(-50, 50))
        p2 = (rng.randint(-50, 50), rng.randint(-50, 50))
        while p2 == p1:
            p2 = (rng.randint(-50, 50), rng.randint(-50, 50))
        manager.add_point(p1)
        manager.add_point(p2)
        
        for step in range(2, 16):
            candidates = manager.get_candidates(rng)
            if not candidates:
                # Fallback to random
                candidates = [(rng.randint(-100, 100), rng.randint(-100, 100)) for _ in range(10)]
                
            best_cand = None
            best_cand_score = -1
            
            for cand in candidates:
                # Check if adding cand creates 5+ collinear
                # We can simulate add
                # To be fast, we just check the lines it would join
                # But manager.add_point modifies state. 
                # We need a dry-run or careful revert.
                # For simplicity and speed, we'll just add and check, then revert if bad.
                # But manager doesn't support easy revert of last add without index shift issues if we remove.
                # Let's just pick the first valid candidate that doesn't have 5.
                
                # Simpler: Just pick a random candidate from the list, 
                # but prioritize those on lines with 3 points?
                # The get_candidates already prioritizes completions.
                
                # Let's just take the first one that is valid (no 5)
                # We'll add it and check.
                s, h5 = manager.add_point(cand)
                if h5:
                    # Revert
                    # Find the index of the point just added
                    # It's the last one
                    idx = len(manager.points) - 1
                    manager.remove_point(idx)
                    continue
                else:
                    best_cand = cand
                    break
                    
            if best_cand is None:
                # Couldn't find a valid candidate? Try random
                while True:
                    cand = (rng.randint(-100, 100), rng.randint(-100, 100))
                    s, h5 = manager.add_point(cand)
                    if not h5:
                        break
                    else:
                        idx = len(manager.points) - 1
                        manager.remove_point(idx)
                        
        constructed_points = [p[:] for p in manager.points]
        
        # Phase 2: Local Search
        score, refined_points = _local_search(constructed_points, rng, deadline, best_score, best_points)
        
        if score > best_score:
            best_score = score
            best_points = refined_points
    
    return best_points
# EVOLVE-BLOCK-END
