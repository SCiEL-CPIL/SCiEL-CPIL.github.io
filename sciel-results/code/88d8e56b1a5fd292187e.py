"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before hard limit


def _compute_score(circles):
    """Compute sum of radii for a valid packing, or -inf if invalid."""
    n = len(circles)
    if n == 0:
        return 0.0

    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Check all radii are positive
    if np.any(r <= 0):
        return -np.inf

    # Check all circles are inside the unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf

    # Check no overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf

    return np.sum(r)


def _force_relax(x, y, r, n, iterations=500, dt=0.01, repulsion_strength=10.0):
    """
    Force-directed relaxation: circles repel when overlapping and are
    attracted toward center of empty space. Returns improved positions.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for it in range(iterations):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]

                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force = repulsion_strength * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist and dist <= 1e-12:
                    # Nearly coincident, push apart randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    force = repulsion_strength * min_dist
                    fx[i] += force * np.cos(angle)
                    fy[i] += force * np.sin(angle)
                    fx[j] -= force * np.cos(angle)
                    fy[j] -= force * np.sin(angle)

        # Wall constraints: push away from boundaries
        for i in range(n):
            # Left wall
            if x[i] - r[i] < 0:
                fx[i] += repulsion_strength * (r[i] - x[i])
            # Right wall
            if x[i] + r[i] > 1:
                fx[i] -= repulsion_strength * (x[i] + r[i] - 1)
            # Bottom wall
            if y[i] - r[i] < 0:
                fy[i] += repulsion_strength * (r[i] - y[i])
            # Top wall
            if y[i] + r[i] > 1:
                fy[i] -= repulsion_strength * (y[i] + r[i] - 1)

        # Update positions with damping
        x += dt * fx
        y += dt * fy

        # Clamp to valid range
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

    return x, y, r


def _make_initial_grid(n):
    """Create initial grid layout with strict non-overlap."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Shrink radius slightly to ensure strict disjoint interiors
            r = 0.45 / side
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _make_initial_random(n, seed=42):
    """Create random initial layout."""
    rng = np.random.RandomState(seed)
    # Place circles at random positions with small radii
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    return np.column_stack([x, y, r])


def _grow_radii_with_relaxation(x, y, r, n, time_budget):
    """
    Iteratively grow radii and use force-directed relaxation to maintain
    valid packing. Returns improved circles and score.
    """
    start_time = time.time()
    x = x.copy()
    y = y.copy()
    r = r.copy()

    # Start with small growth increments
    growth_factor = 0.005
    relaxation_iters = 100

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break

        # Try to grow all radii by a small amount
        r_new = r * (1 + growth_factor)

        # Check if growth is feasible (rough estimate)
        # Force relaxation to fix overlaps
        x_rel, y_rel, r_rel = _force_relax(x, y, r_new, n, iterations=relaxation_iters)

        # Validate
        circles = np.column_stack([x_rel, y_rel, r_rel])
        score = _compute_score(circles)

        if not np.isinf(score) and score > np.sum(r):
            # Accept the growth
            x, y, r = x_rel, y_rel, r_rel
            # Slow down growth as we get larger
            growth_factor *= 0.95
        else:
            # Growth not feasible, try smaller increment
            growth_factor *= 0.8

        if growth_factor < 1e-6:
            break

    circles = np.column_stack([x, y, r])
    score = _compute_score(circles)
    if np.isinf(score):
        return None
    return circles, score


def _optimize_from_initial(initial, n, time_budget):
    """Run optimization from a given initial configuration."""
    start_time = time.time()

    try:
        x = initial[:, 0].copy()
        y = initial[:, 1].copy()
        r = initial[:, 2].copy()

        # First, do a force relaxation to ensure valid starting position
        x, y, r = _force_relax(x, y, r, n, iterations=50)

        # Then grow radii with relaxation
        result = _grow_radii_with_relaxation(x, y, r, n, time_budget * 0.7)

        if result is not None:
            circles, score = result

            # Final polish: more relaxation iterations
            x_pol = circles[:, 0].copy()
            y_pol = circles[:, 1].copy()
            r_pol = circles[:, 2].copy()

            remaining = time_budget - (time.time() - start_time)
            if remaining > 2:
                x_pol, y_pol, r_pol = _force_relax(x_pol, y_pol, r_pol, n, iterations=200)
                circles_pol = np.column_stack([x_pol, y_pol, r_pol])
                score_pol = _compute_score(circles_pol)
                if not np.isinf(score_pol) and score_pol > score:
                    circles, score = circles_pol, score_pol

            return circles, score

    except Exception:
        return None


def construct():
    """Construct 26 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    best_circles = None
    best_score = -np.inf

    # Try multiple initial configurations
    initial_configs = []

    # Grid initial
    initial_configs.append(_make_initial_grid(N))

    # Random initials with different seeds
    for seed in range(8):
        initial_configs.append(_make_initial_random(N, seed=seed + 100))

    # Run optimization from each initial
    for i, init in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()

        if remaining < 3.0:  # Need at least 3 seconds for one optimization
            break

        # Allocate time budget based on remaining time
        time_budget = min(remaining * 0.85, 15.0)

        result = _optimize_from_initial(init, N, time_budget)

        if result is not None:
            circles, score = result
            if score > best_score:
                best_score = score
                best_circles = circles.copy()

        # Check if we should stop
        if time.time() > deadline:
            break

    # If no valid result, fall back to grid
    if best_circles is None:
        best_circles = _make_initial_grid(N)

    # Final validation
    final_score = _compute_score(best_circles)
    if np.isinf(final_score):
        # As a last resort, return a very safe small circle configuration
        best_circles = _make_initial_grid(N)

    return best_circles
