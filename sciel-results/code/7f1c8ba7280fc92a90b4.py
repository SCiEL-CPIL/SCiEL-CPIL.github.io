"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from itertools import combinations
import time
import math

# EVOLVE-BLOCK-START
def construct():
    N = 30
    MIN_COORD = -30
    MAX_COORD = 30
    TIME_LIMIT = 55  # Leave 5s buffer

    def calculate_score(pts):
        """
        Returns the maximum multiplicity of a single positive doubled area.
        """
        if len(pts) < 3:
            return 0
        
        area_counts = {}
        
        # Pre-extract coordinates for speed
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        for i in range(N):
            x1, y1 = xs[i], ys[i]
            for j in range(i + 1, N):
                x2, y2 = xs[j], ys[j]
                for k in range(j + 1, N):
                    x3, y3 = xs[k], ys[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if area > 0:
                        area_counts[area] = area_counts.get(area, 0) + 1
        
        if not area_counts:
            return 0
        return max(area_counts.values())

    def is_valid_point(x, y, occupied_set):
        return MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD and (x, y) not in occupied_set

    def hill_climb(pts, time_budget):
        """
        Greedy Hill Climbing with random moves.
        """
        current_pts = [tuple(p) for p in pts]
        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        occupied = set(current_pts)
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 1000 # Allow more exploration before stopping
        
        while time.time() - start_time < time_budget and no_improve_count < max_no_improve:
            idx = random.randint(0, N - 1)
            old_pt = current_pts[idx]
            
            # Bias towards local moves for refinement
            if random.random() < 0.8:
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                nx, ny = old_pt[0] + dx, old_pt[1] + dy
            else:
                nx = random.randint(MIN_COORD, MAX_COORD)
                ny = random.randint(MIN_COORD, MAX_COORD)
            
            # Check if new position is valid (in bounds and not occupied by another point)
            if not is_valid_point(nx, ny, occupied - {old_pt}):
                no_improve_count += 1
                continue
            
            # Apply move
            occupied.remove(old_pt)
            occupied.add((nx, ny))
            current_pts[idx] = (nx, ny)
            
            new_score = calculate_score(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts[:]
                no_improve_count = 0
            else:
                # Revert
                occupied.remove((nx, ny))
                occupied.add(old_pt)
                current_pts[idx] = old_pt
                no_improve_count += 1
                
        return best_pts, best_score

    def generate_grid(r, c, skew_x=0, skew_y=0, scale_x=1, scale_y=1):
        """
        Generate a grid of r x c points.
        """
        pts = []
        start_x = -((c - 1) * scale_x) // 2
        start_y = -((r - 1) * scale_y) // 2
        
        for i in range(c):
            for j in range(r):
                x = start_x + i * scale_x + j * skew_x
                y = start_y + j * scale_y + i * skew_y
                if MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD:
                    pts.append((x, y))
        return pts

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Standard Grids (Factors of 30: 1x30, 2x15, 3x10, 5x6, 6x5, 10x3, 15x2, 30x1)
    for r in [1, 2, 3, 5, 6, 10, 15, 30]:
        c = 30 // r
        # Try different scales and skews
        for scale_x in [1, 2, 3]:
            for scale_y in [1, 2, 3]:
                for skew_x in [-2, -1, 0, 1, 2]:
                    for skew_y in [-2, -1, 0, 1, 2]:
                        pts = generate_grid(r, c, skew_x, skew_y, scale_x, scale_y)
                        if len(pts) == N and len(set(pts)) == N:
                            candidates.append(pts)
    
    # Random Points
    for _ in range(10):
        pts = []
        occupied = set()
        while len(pts) < N:
            x = random.randint(MIN_COORD, MAX_COORD)
            y = random.randint(MIN_COORD, MAX_COORD)
            if (x, y) not in occupied:
                pts.append((x, y))
                occupied.add((x, y))
        candidates.append(pts)
    
    # Evaluate candidates and pick the best few to run Hill Climbing on
    scored_candidates = []
    for pts in candidates:
        if len(set(pts)) != N:
            continue
        score = calculate_score(pts)
        scored_candidates.append((score, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_final_pts = None
    best_final_score = -1
    
    # Run Hill Climbing on the top 3 initial candidates
    time_per_run = TIME_LIMIT / 5
    start_total = time.time()
    
    for i, (init_score, init_pts) in enumerate(scored_candidates[:3]):
        if time.time() - start_total > TIME_LIMIT - 5:
            break
        remaining = TIME_LIMIT - (time.time() - start_total)
        pts, score = hill_climb(init_pts, remaining)
        if score > best_final_score:
            best_final_score = score
            best_final_pts = pts
            
    # If we have time, try a random start
    if time.time() - start_total < TIME_LIMIT - 10:
        remaining = TIME_LIMIT - (time.time() - start_total)
        # Pick a random candidate from the middle of the list
        if len(scored_candidates) > 3:
            random_idx = random.randint(3, len(scored_candidates) - 1)
            _, random_pts = scored_candidates[random_idx]
        else:
            random_pts = scored_candidates[-1][1]
            
        pts, score = hill_climb(random_pts, remaining)
        if score > best_final_score:
            best_final_score = score
            best_final_pts = pts
            
    if best_final_pts:
        return best_final_pts
        
    # Fallback
    return candidates[0] if candidates else [(i, 0) for i in range(N)]
# EVOLVE-BLOCK-END
