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
    TIME_LIMIT = 50  # Leave 10s buffer

    def calculate_scores(pts):
        """
        Returns a tuple (max_mult, counts_dict) for the given points.
        """
        counts = {}
        # Unpack points for faster access
        # pts is a list of tuples
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i + 1, N):
                x2, y2 = pts[j]
                for k in range(j + 1, N):
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if area > 0:
                        counts[area] = counts.get(area, 0) + 1
        
        if not counts:
            return 0, counts
        max_mult = max(counts.values())
        return max_mult, counts

    def get_neighbors(pts, num_neighbors=1):
        """
        Generate neighbor configurations by moving random points.
        """
        neighbors = []
        occupied = set(pts)
        for _ in range(num_neighbors):
            new_pts = pts[:]
            idx = random.randint(0, N - 1)
            # Try to find a new position
            # Random walk or random jump?
            # Random jump is better for global exploration, local for refinement.
            # Let's use a mix.
            if random.random() < 0.5:
                # Local move
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                nx = pts[idx][0] + dx
                ny = pts[idx][1] + dy
                if MIN_COORD <= nx <= MAX_COORD and MIN_COORD <= ny <= MAX_COORD and (nx, ny) not in occupied:
                    new_pts[idx] = (nx, ny)
                    neighbors.append(new_pts)
            else:
                # Global jump
                while True:
                    nx = random.randint(MIN_COORD, MAX_COORD)
                    ny = random.randint(MIN_COORD, MAX_COORD)
                    if (nx, ny) not in occupied:
                        new_pts[idx] = (nx, ny)
                        neighbors.append(new_pts)
                        break
        return neighbors

    def simulated_annealing(initial_pts, time_budget):
        current_pts = initial_pts[:]
        current_score, _ = calculate_scores(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        
        start_time = time.time()
        temperature = 10.0
        cooling_rate = 0.999
        max_iter = 0
        
        while time.time() - start_time < time_budget:
            max_iter += 1
            neighbors = get_neighbors(current_pts, 1)
            if not neighbors:
                continue
            next_pts = neighbors[0]
            next_score, _ = calculate_scores(next_pts)
            
            delta = next_score - current_score
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_pts = next_pts
                current_score = next_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts[:]
            
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
                
        return best_pts, best_score

    # Generate initial candidates
    candidates = []
    
    # 1. Rectangular grids (from seed logic, but we will optimize them)
    factors = [1, 2, 3, 5, 6, 10, 15, 30]
    for r in factors:
        c = 30 // r
        # Generate grid
        shift_x = -(c - 1) // 2
        shift_y = -(r - 1) // 2
        pts_grid = [(x + shift_x, y + shift_y) for x in range(c) for y in range(r)]
        # Check bounds
        if all(MIN_COORD <= p[0] <= MAX_COORD and MIN_COORD <= p[1] <= MAX_COORD for p in pts_grid):
            candidates.append(pts_grid)
        if r != c:
            shift_x2 = -(r - 1) // 2
            shift_y2 = -(c - 1) // 2
            pts_grid2 = [(x + shift_x2, y + shift_y2) for x in range(r) for y in range(c)]
            if all(MIN_COORD <= p[0] <= MAX_COORD and MIN_COORD <= p[1] <= MAX_COORD for p in pts_grid2):
                candidates.append(pts_grid2)

    # 2. Random initializations
    for _ in range(5):
        random_pts = []
        occupied = set()
        while len(random_pts) < N:
            x = random.randint(MIN_COORD, MAX_COORD)
            y = random.randint(MIN_COORD, MAX_COORD)
            if (x, y) not in occupied:
                random_pts.append((x, y))
                occupied.add((x, y))
        candidates.append(random_pts)

    # Evaluate initial candidates to pick a good starting point for SA
    best_initial_pts = None
    best_initial_score = -1
    for pts in candidates:
        if len(set(pts)) != N:
            continue
        score, _ = calculate_scores(pts)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_pts = pts

    # Run Simulated Annealing on the best initial candidate
    # We can also run it on a few others if time permits, but one good run is better.
    # Let's run SA on the best grid found.
    if best_initial_pts:
        final_pts, final_score = simulated_annealing(best_initial_pts, TIME_LIMIT)
        return final_pts
    
    # Fallback
    return candidates[0] if candidates else [(i, 0) for i in range(N)]
# EVOLVE-BLOCK-END
