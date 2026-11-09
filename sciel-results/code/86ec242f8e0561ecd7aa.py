import numpy as np
import random

def squares_in_square10() -> tuple:
    n = 10
    # Initial guess: 1 large, 9 small
    # Large square at top-right
    L = 0.1
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Large square
    centers[9] = [1.0 - L/2, 1.0 - L/2]
    sides[9] = L
    
    # 9 small squares in a 3x3 grid initially
    s = (1.0 - L) / 3.0
    for i in range(9):
        row = i // 3
        col = i % 3
        cx = (col + 0.5) * s
        cy = (row + 0.5) * s
        centers[i] = [cx, cy]
        sides[i] = s
        
    # Local Search
    # We will try to improve the sum of sides.
    # We fix the large square's position (top-right) but allow its size to change?
    # Let's allow all squares to move and resize, but with a bias.
    
    def is_valid(centers, sides):
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0: return False
            if cx - s/2 < -1e-7 or cx + s/2 > 1 + 1e-7: return False
            if cy - s/2 < -1e-7 or cy + s/2 > 1 + 1e-7: return False
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                t = (s_i + s_j)/2 - 1e-7
                if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                    return False
        return True

    def get_score(centers, sides):
        return np.sum(sides)

    current_score = get_score(centers, sides)
    
    # Simple hill climbing
    # To avoid getting stuck, we can try a few random restarts or perturbations.
    # Given 600s, we can run for ~10-20 seconds.
    
    import time
    start_time = time.time()
    timeout = 10.0 # seconds
    
    while time.time() - start_time < timeout:
        # Pick a random square
        idx = random.randint(0, n-1)
        # Perturb center and side
        new_centers = centers.copy()
        new_sides = sides.copy()
        
        # Small random changes
        new_centers[idx] += np.random.normal(0, 0.01, 2)
        new_sides[idx] *= (1 + np.random.normal(0, 0.01))
        
        # Ensure side > 0
        new_sides[idx] = max(new_sides[idx], 1e-6)
        
        if is_valid(new_centers, new_sides):
            new_score = get_score(new_centers, new_sides)
            if new_score > current_score:
                centers = new_centers
                sides = new_sides
                current_score = new_score
        # Occasionally, try a larger move or reset one square
        if random.random() < 0.01:
             # Reset a random small square to a grid position
             idx = random.randint(0, 8)
             # This is a bit complex to implement safely in a loop without knowing L
             pass

    return centers, sides
