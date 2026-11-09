"""Improved program: Structural change to incremental collinearity tracking with dense initialization
and simulated annealing. This allows O(N) move evaluations instead of O(N^3), enabling significantly
more iterations within the time budget. The goal remains to maximize collinear triples to minimize
the largest 3-free subset."""
import random
import math
import time
from collections import defaultdict

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0  # seconds (evaluator kills at 150s, keep safety margin)

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    # p1, p2, p3 are tuples (x, y)
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _get_line_key(p1, p2):
    """Return a canonical key for the line passing through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return (0, p1[0])
    if dy == 0:
        return (p1[1], 0)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonicalize direction
    if dx < 0:
        dx = -dx
        dy = -dy
    # Key is (slope, intercept) but using (dy, dx, c) to avoid float issues
    # Line: dy*x - dx*y = c
    c = dy * p1[0] - dx * p1[1]
    return (dy, dx, c)

def _count_three_collinear(points):
    """Count the number of triples of collinear points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    # Use line counting for speed
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            line_counts[key] += 1
            if line_counts[key] >= 3: # 3 pairs means 4 points? No.
                # If 4 points are on a line, there are C(4,2)=6 pairs.
                # If 3 points, C(3,2)=3 pairs.
                # So if count >= 6, we have at least 4 points.
                # Wait, C(4,2)=6. C(5,2)=10.
                # So if line_counts[key] >= 6, we have 4+ collinear.
                pass
    for count in line_counts.values():
        if count >= 6:
            return True
    return False

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear.
    points: list of tuples (x, y). Returns the size of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity pairs
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinearity[i][j] = mask
            collinearity[j][i] = mask
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinearity[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

class LineTracker:
    def __init__(self, points):
        self.points = points
        self.line_counts = defaultdict(int)
        self._rebuild()
        
    def _rebuild(self):
        self.line_counts.clear()
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                key = _get_line_key(self.points[i], self.points[j])
                self.line_counts[key] += 1
                
    def get_triple_count(self):
        """Calculate number of collinear triples from line pair counts.
        If a line has k points, it contributes C(k, 3) to the triple count.
        We have pair counts. If a line has k points, it has C(k, 2) pairs.
        So we need to map pair counts back to point counts?
        No, this is tricky. 
        Alternative: Just count triples directly in O(N^3) but that's slow.
        
        Let's stick to a simpler metric for SA: The number of collinear pairs is not enough.
        We want to maximize triples.
        
        Actually, we can just use the O(N^3) count if we make it fast with numpy or just accept it's slower?
        No, the directive asks for structural change.
        
        Let's use the number of "bad" lines (lines with >=3 points) as a proxy?
        Or just count the sum of C(k,3) where k is the number of points on the line.
        To get k, we can't just look at pair counts easily if lines overlap?
        Lines are unique. 
        For each line key, the number of points k is determined by C(k,2) = count.
        k = (1 + sqrt(1 + 8*count)) / 2.
        """
        total_triples = 0
        for count in self.line_counts.values():
            # Solve k(k-1)/2 = count for k
            # k^2 - k - 2*count = 0
            # k = (1 + sqrt(1 + 8*count)) / 2
            d = 1 + 8 * count
            root = math.isqrt(d)
            if root * root == d:
                k = (1 + root) // 2
                if k >= 3:
                    total_triples += k * (k - 1) * (k - 2) // 6
        return total_triples

    def has_four_collinear(self):
        for count in self.line_counts.values():
            # C(k,2) >= 6 => k >= 4
            if count >= 6:
                return True
        return False

    def move_point(self, idx, new_point):
        """Move point idx to new_point and update line counts."""
        old_point = self.points[idx]
        if old_point == new_point:
            return
        
        # Remove old point contributions
        for j in range(len(self.points)):
            if j == idx:
                continue
            key = _get_line_key(old_point, self.points[j])
            self.line_counts[key] -= 1
            if self.line_counts[key] == 0:
                del self.line_counts[key]
        
        # Add new point contributions
        self.points[idx] = new_point
        for j in range(len(self.points)):
            if j == idx:
                continue
            key = _get_line_key(new_point, self.points[j])
            self.line_counts[key] += 1

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Initialize with dense random points to encourage collinearities
    # Box [-10, 10] gives 21x21 = 441 spots for 20 points.
    points = []
    for _ in range(20):
        while True:
            p = (random.randint(-10, 10), random.randint(-10, 10))
            if p not in points:
                points.append(p)
                break
    
    # Ensure no four collinear initially
    tracker = LineTracker(points)
    attempts = 0
    while tracker.has_four_collinear() and attempts < 1000:
        idx = random.randint(0, 19)
        old = points[idx]
        new = (random.randint(-10, 10), random.randint(-10, 10))
        if new not in points:
            tracker.move_point(idx, new)
        attempts += 1
        
    if tracker.has_four_collinear():
        # Fallback: spread out
        points = []
        for _ in range(20):
            while True:
                p = (random.randint(-50, 50), random.randint(-50, 50))
                if p not in points:
                    points.append(p)
                    break
        tracker = LineTracker(points)
        while tracker.has_four_collinear():
            idx = random.randint(0, 19)
            new = (random.randint(-50, 50), random.randint(-50, 50))
            if new not in points:
                tracker.move_point(idx, new)

    current_triples = tracker.get_triple_count()
    best_triples = current_triples
    best_points = list(points)
    best_exact_score = _max_3_free_subset(points)
    
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.1
    
    while temperature > min_temperature and _time_left() > 15.0:
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Jitter or random move?
        # Jitter is better for local refinement
        if random.random() < 0.7:
            new_point = (old_point[0] + random.randint(-2, 2), old_point[1] + random.randint(-2, 2))
        else:
            new_point = (random.randint(-15, 15), random.randint(-15, 15))
            
        if new_point in points or new_point[0] < -100 or new_point[0] > 100 or new_point[1] < -100 or new_point[1] > 100:
            continue
            
        # Try move
        tracker.move_point(idx, new_point)
        
        if tracker.has_four_collinear():
            tracker.move_point(idx, old_point)
            continue
            
        new_triples = tracker.get_triple_count()
        
        delta = new_triples - current_triples
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = list(points)
                # Update exact score periodically or on improvement
                if _time_left() > 20.0:
                    exact_score = _max_3_free_subset(points)
                    if exact_score < best_exact_score:
                        best_exact_score = exact_score
        else:
            tracker.move_point(idx, old_point)
            
        temperature *= cooling_rate
    
    # Final refinement
    points = list(best_points)
    current_exact = _max_3_free_subset(points)
    
    for _ in range(200):
        if _time_left() < 5.0:
            break
        idx = random.randint(0, 19)
        old_point = points[idx]
        new_point = (old_point[0] + random.randint(-2, 2), old_point[1] + random.randint(-2, 2))
        if new_point in points or new_point[0] < -100 or new_point[0] > 100 or new_point[1] < -100 or new_point[1] > 100:
            continue
        points[idx] = new_point
        if _has_four_collinear(points):
            points[idx] = old_point
            continue
        new_exact = _max_3_free_subset(points)
        if new_exact <= current_exact:
            current_exact = new_exact
            if current_exact < best_exact_score:
                best_exact_score = current_exact
                best_points = list(points)
        else:
            points[idx] = old_point
    
    return [list(p) for p in best_points]
