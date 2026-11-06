import time
import random
import sys
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def count_triples(points):
        line_map = build_line_map(points)
        count = 0
        for line_pts in line_map.values():
            if len(line_pts) == 3:
                count += 1
        return count

    def triple_structure_score(points):
        """Score: number of 3-point lines. Higher = more collinearity = better."""
        line_map = build_line_map(points)
        score = 0
        for line_pts in line_map.values():
            if len(line_pts) == 3:
                score += 1
        return score

    def calculate_exact_score(points):
        n = len(points)
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) == 3:
                pts = sorted(line_pts)
                triples.append((pts[0], pts[1], pts[2]))
        
        if not triples:
            return n

        conflict_mask = [0] * n
        for i, j, k in triples:
            conflict_mask[i] |= (1 << j) | (1 << k)
            conflict_mask[j] |= (1 << i) | (1 << k)
            conflict_mask[k] |= (1 << i) | (1 << j)

        best = [0]
        
        def greedy_upper_bound(mask):
            remaining = mask
            count = 0
            while remaining:
                best_v = -1
                best_deg = -1
                tmp = remaining
                while tmp:
                    bit = tmp & -tmp
                    v = bit.bit_length() - 1
                    deg = (conflict_mask[v] & remaining).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    tmp ^= bit
                if best_v < 0:
                    break
                count += 1
                remaining &= ~(1 << best_v)
                remaining &= ~conflict_mask[best_v]
            return count

        def backtrack(available_mask, current_size):
            if current_size + available_mask.bit_count() <= best[0]:
                return
            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            ub = greedy_upper_bound(available_mask)
            if current_size + ub <= best[0]:
                return
            best_v = -1
            best_deg = -1
            tmp = available_mask
            while tmp:
                bit = tmp & -tmp
                v = bit.bit_length() - 1
                deg = (conflict_mask[v] & available_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                tmp ^= bit
            v_bit = 1 << best_v
            new_available = available_mask ^ v_bit
            new_available &= ~conflict_mask[best_v]
            backtrack(new_available, current_size + 1)
            backtrack(available_mask ^ v_bit, current_size)
        
        backtrack((1 << n) - 1, 0)
        return best[0]

    def greedy_construct(seed):
        """Greedy sequential construction with given random seed."""
        rng = random.Random(seed)
        pts = []
        existing_lines = defaultdict(set)  # line_key -> set of point indices
        
        for step in range(N):
            best_pos = None
            best_score = -1
            best_new_triples = -1
            
            # Generate candidate positions
            candidates = set()
            # Random candidates
            for _ in range(500):
                x = rng.randint(COORD_MIN, COORD_MAX)
                y = rng.randint(COORD_MIN, COORD_MAX)
                candidates.add((x, y))
            # Also try positions that create 3-point lines
            # For each pair of existing points, find third point on same line
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Extend line in both directions
                    for sign in [1, -1]:
                        for t in range(1, 11):
                            nx = x2 + sign * t * dx
                            ny = y2 + sign * t * dy
                            if COORD_MIN <= nx <= COORD_MAX and COORD_MIN <= ny <= COORD_MAX:
                                candidates.add((nx, ny))
                            nx = x1 - sign * t * dx
                            ny = y1 - sign * t * dy
                            if COORD_MIN <= nx <= COORD_MAX and COORD_MIN <= ny <= COORD_MAX:
                                candidates.add((nx, ny))
            
            # Filter candidates
            valid_candidates = []
            for (x, y) in candidates:
                if (x, y) in [(px, py) for px, py in pts]:
                    continue
                # Check no 4 collinear
                line_counts = defaultdict(int)
                for i in range(len(pts)):
                    key = line_key((x, y), pts[i])
                    line_counts[key] += 1
                if any(c >= 3 for c in line_counts.values()):
                    continue
                valid_candidates.append((x, y))
            
            if not valid_candidates:
                # Fallback: just pick a random valid position
                for _ in range(1000):
                    x = rng.randint(COORD_MIN, COORD_MAX)
                    y = rng.randint(COORD_MIN, COORD_MAX)
                    if (x, y) in [(px, py) for px, py in pts]:
                        continue
                    line_counts = defaultdict(int)
                    for i in range(len(pts)):
                        key = line_key((x, y), pts[i])
                        line_counts[key] += 1
                    if any(c >= 3 for c in line_counts.values()):
                        continue
                    valid_candidates.append((x, y))
                    break
                if not valid_candidates:
                    return None
            
            # Score each candidate
            for (x, y) in valid_candidates:
                # Count new 3-point lines formed
                new_triples = 0
                line_counts = defaultdict(list)
                for i in range(len(pts)):
                    key = line_key((x, y), pts[i])
                    line_counts[key].append(i)
                for key, idxs in line_counts.items():
                    if len(idxs) == 2:
                        new_triples += 1
                # Also check if this creates a 4-point line (shouldn't due to filter)
                for key, idxs in line_counts.items():
                    if len(idxs) >= 3:
                        new_triples = -100  # penalty
                        break
                
                if new_triples > best_score:
                    best_score = new_triples
                    best_pos = (x, y)
            
            if best_pos is None:
                return None
            
            pts.append([best_pos[0], best_pos[1]])
        
        return pts

    def local_improve(pts, rng, max_iters=200):
        """Local search to improve triple structure score."""
        current = [p[:] for p in pts]
        current_score = triple_structure_score(current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            idx = rng.randint(0, N - 1)
            old = current[idx][:]
            
            # Try moving to a position that increases triple count
            best_new = None
            best_new_score = current_score - 1  # must improve
            
            # Candidate positions
            candidates = set()
            for _ in range(200):
                x = rng.randint(COORD_MIN, COORD_MAX)
                y = rng.randint(COORD_MIN, COORD_MAX)
                candidates.add((x, y))
            # Extend lines through other points
            for i in range(N):
                if i == idx:
                    continue
                x1, y1 = current[i]
                x0, y0 = old
                dx = x1 - x0
                dy = y1 - y0
                for sign in [1, -1]:
                    for t in range(1, 6):
                        nx = x1 + sign * t * dx
                        ny = y1 + sign * t * dy
                        if COORD_MIN <= nx <= COORD_MAX and COORD_MIN <= ny <= COORD_MAX:
                            candidates.add((nx, ny))
            
            for (x, y) in candidates:
                if (x, y) == (old[0], old[1]):
                    continue
                if any(current[i] == [x, y] for i in range(N) if i != idx):
                    continue
                # Check validity
                current[idx] = [x, y]
                valid = True
                line_counts = defaultdict(int)
                for i in range(N):
                    if i == idx:
                        continue
                    key = line_key((x, y), current[i])
                    line_counts[key] += 1
                if any(c >= 3 for c in line_counts.values()):
                    valid = False
                if not valid:
                    current[idx] = old
                    continue
                new_score = triple_structure_score(current)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new = [x, y]
                current[idx] = old
            
            if best_new is not None:
                current[idx] = best_new
                current_score = best_new_score
            else:
                # No improvement, try a random valid move
                for _ in range(50):
                    x = rng.randint(COORD_MIN, COORD_MAX)
                    y = rng.randint(COORD_MIN, COORD_MAX)
                    if any(current[i] == [x, y] for i in range(N) if i != idx):
                        continue
                    current[idx] = [x, y]
                    line_counts = defaultdict(int)
                    for i in range(N):
                        if i == idx:
                            continue
                        key = line_key((x, y), current[i])
                        line_counts[key] += 1
                    if any(c >= 3 for c in line_counts.values()):
                        current[idx] = old
                        continue
                    new_score = triple_structure_score(current)
                    if new_score >= current_score - 2:  # accept slightly worse
                        current_score = new_score
                    else:
                        current[idx] = old
                    break
        
        return current

    # Main: try multiple greedy constructions with different seeds
    best_config = None
    best_score = 20
    
    # Try a range of seeds
    seeds = list(range(50))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        pts = greedy_construct(seed)
        if pts is None:
            continue
        
        if not is_valid(pts):
            continue
        
        # Local improvement
        rng = random.Random(seed + 1000)
        pts = local_improve(pts, rng, max_iters=100)
        
        if not is_valid(pts):
            continue
        
        exact = calculate_exact_score(pts)
        if exact < best_score:
            best_score = exact
            best_config = [p[:] for p in pts]
            print(f"Seed {seed}: score {exact}", file=sys.stderr)
        
        if best_score <= 7:
            break
    
    if best_config is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_config

if __name__ == "__main__":
    res = construct()
    print(res)
