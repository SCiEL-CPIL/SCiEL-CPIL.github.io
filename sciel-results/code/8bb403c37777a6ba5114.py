"""Program to find 32 points on triangular lattice maximizing unit-distance edges."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit

    # Triangular lattice distance squared:
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # |z1 - z2|^2 = da^2 + db^2 + da*db where da = a1-a2, db = b1-b2
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def validate(points):
        """Check all constraints: 32 distinct points, |a|,|b| <= 64, mutual distance >= 1."""
        if len(points) != 32:
            return False
        seen = set()
        for p in points:
            if len(p) != 2:
                return False
            a, b = p
            if not isinstance(a, int) or not isinstance(b, int):
                return False
            if abs(a) > 64 or abs(b) > 64:
                return False
            if (a, b) in seen:
                return False
            seen.add((a, b))
        # Check mutual distance at least 1 (dist_sq >= 1 for all pairs)
        for i in range(32):
            for j in range(i + 1, 32):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True

    def hex_spiral(n):
        """Generate n points in a hexagonal spiral pattern on triangular lattice."""
        points = [(0, 0)]
        if n == 1:
            return points

        # Directions in the 6-fold symmetric triangular lattice
        # The 6 neighbors of (a,b) are at:
        # (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

        # Build hexagonal layers
        # Layer k has 6k points
        layer = 1
        while len(points) < n:
            # Start at the top of the current layer
            # For layer k, start at (k, 0) and go around
            # Actually, let's use a different approach:
            # Layer 1: 6 points around origin
            # Layer 2: 12 points around layer 1
            # etc.

            # Position to start this layer
            # Go from (0,0) in direction 0 for 'layer' steps
            ax, ay = 0, 0
            for _ in range(layer):
                ax += directions[0][0]
                ay += directions[0][1]

            # Now walk around the hexagon
            for side in range(6):
                dx, dy = directions[side]
                for step in range(layer):
                    if len(points) >= n:
                        break
                    points.append((ax, ay))
                    ax += dx
                    ay += dy
            layer += 1

        return points[:n]

    def random_search(points, num_moves, temperature):
        """Simulated annealing style local search."""
        best_points = list(points)
        best_score = count_edges(best_points)
        current_points = list(points)
        current_score = best_score

        n = len(current_points)

        # Precompute neighbor offsets for unit distance
        # dist_sq == 1 when da^2 + db^2 + da*db == 1
        # Solutions: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
        unit_neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

        for move in range(num_moves):
            if time.time() > deadline:
                break

            # Pick a random point to move
            idx = random.randint(0, n - 1)
            old_pos = current_points[idx]

            # Try a random new position nearby
            # Random offset in range [-3, 3] for both coordinates
            da = random.randint(-4, 4)
            db = random.randint(-4, 4)
            new_pos = (old_pos[0] + da, old_pos[1] + db)

            # Check bounds
            if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                continue

            # Check if new position is already occupied
            if new_pos in [current_points[i] for i in range(n) if i != idx]:
                continue

            # Check minimum distance constraint
            valid = True
            for i in range(n):
                if i == idx:
                    continue
                if dist_sq(new_pos, current_points[i]) < 1:
                    valid = False
                    break

            if not valid:
                continue

            # Compute new score
            new_score = current_score
            # Remove edges involving old_pos
            for i in range(n):
                if i == idx:
                    continue
                if dist_sq(old_pos, current_points[i]) == 1:
                    new_score -= 1
            # Add edges involving new_pos
            for i in range(n):
                if i == idx:
                    continue
                if dist_sq(new_pos, current_points[i]) == 1:
                    new_score += 1

            # Accept if better or with some probability (simulated annealing)
            if new_score >= current_score:
                current_points[idx] = new_pos
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
            else:
                # Accept worse moves with decreasing probability
                delta = current_score - new_score
                prob = math.exp(-delta / max(temperature, 0.01))
                if random.random() < prob:
                    current_points[idx] = new_pos
                    current_score = new_score

            # Cool down
            temperature *= 0.9999

        return best_points, best_score

    # Generate initial hexagonal spiral
    initial_points = hex_spiral(32)
    best_points = list(initial_points)
    best_score = count_edges(best_points)

    # Try multiple random restarts with local search
    random.seed(42)
    num_restarts = 20
    num_moves = 5000

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            # Use the hex spiral
            start_points = list(initial_points)
        else:
            # Generate a slightly perturbed hex spiral
            start_points = list(initial_points)
            for i in range(32):
                if random.random() < 0.3:
                    da = random.randint(-2, 2)
                    db = random.randint(-2, 2)
                    new_pos = (start_points[i][0] + da, start_points[i][1] + db)
                    if abs(new_pos[0]) <= 64 and abs(new_pos[1]) <= 64:
                        start_points[i] = new_pos

            # Ensure validity
            if not validate(start_points):
                # Regenerate
                start_points = list(initial_points)

        temperature = 2.0
        result_points, result_score = random_search(start_points, num_moves, temperature)

        if result_score > best_score:
            best_score = result_score
            best_points = result_points

        # Early exit if we reached the theoretical optimum
        if best_score >= 76:
            break

    # Convert to list of lists
    result = [[p[0], p[1]] for p in best_points]

    # Final validation
    if not validate(result):
        # Fallback to hex spiral
        result = [[p[0], p[1]] for p in initial_points]

    return result

# EVOLVE-BLOCK-END
