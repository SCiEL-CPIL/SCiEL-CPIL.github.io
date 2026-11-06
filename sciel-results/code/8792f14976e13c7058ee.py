"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    # Optimal known configuration: 15 points on y=0 (x=0..14), 15 on y=1 (x=0..14)
    # This yields doubled area 1 for 420 triangles.
    seed_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        counts = Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if area2 > 0:
                        counts[area2] += 1
        if not counts:
            return 0
        return max(counts.values())

    best_points = seed_points
    best_score = evaluate(seed_points)  # Should be 420

    # Quick local search: try small perturbations to see if we can exceed 420
    import random
    random.seed(123)

    # Limit search to stay within time budget
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        # Mutate one point
        new_points = list(best_points)
        idx = random.randint(0, 29)
        old_x, old_y = new_points[idx]
        # Try nearby moves first for efficiency
        new_x = old_x + random.randint(-3, 3)
        new_y = old_y + random.randint(-3, 3)
        new_x = max(-30, min(30, new_x))
        new_y = max(-30, min(30, new_y))
        new_points[idx] = (new_x, new_y)

        # Check distinctness
        if len(set(new_points)) != 30:
            continue

        score = evaluate(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    return best_points
# EVOLVE-BLOCK-END
