"""Rewritten program: TSP constructive heuristic using Delaunay-style circumradius scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Strategy: Delaunay-style circumradius heuristic.
    For each candidate u in unvisited_nodes, form the triangle
    (current, destination, u) and compute its circumradius R.
    Smaller R means the three points lie on a small circle, i.e. the
    turn at u is geometrically "tight" and preserves local convexity,
    avoiding long inefficient chords. We combine R with the edge length
    d(current, u) so we still prefer nearby candidates among similarly
    tight triangles:
        score(u) = alpha * R + (1 - alpha) * d(current, u)
    """
    n = len(unvisited_nodes)

    if n == 0:
        return -1

    if n == 1:
        return int(unvisited_nodes[0])

    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # 1. Edge lengths from current to each candidate
    d_curr = distance_matrix[current_node, unvisited_nodes]  # shape (n,)

    # 2. Edge lengths from each candidate to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n,)

    # 3. Edge length current -> destination (constant for all candidates)
    d_cd = distance_matrix[current_node, destination_node]

    # 4. Compute circumradius of triangle (current, destination, u_i) for each i.
    #    For a triangle with side lengths a, b, c and area A, the circumradius is
    #        R = (a * b * c) / (4 * A)
    #    where a = d(current, u_i), b = d(u_i, destination), c = d(current, destination).
    #    Area via Heron's formula: A = sqrt(s*(s-a)*(s-b)*(s-c)), s = (a+b+c)/2.
    #    We guard against degenerate triangles (collinear points) by clamping A.

    a = d_curr          # (n,)
    b = d_dest          # (n,)
    c = np.full(n, d_cd, dtype=np.float64)

    s = (a + b + c) / 2.0
    # Heron's term; clamp to avoid negative due to floating point
    heron = s * (s - a) * (s - b) * (s - c)
    heron = np.maximum(heron, 0.0)
    area = np.sqrt(heron)

    # Circumradius; avoid division by zero for degenerate (collinear) cases
    denom = 4.0 * area
    # Where area is ~0, the triangle is degenerate; treat R as large (penalize)
    # Use a small epsilon to avoid inf, and cap R to a reasonable maximum
    R = np.where(denom > 1e-12, (a * b * c) / np.maximum(denom, 1e-12), np.inf)
    # Cap extremely large circumradii to keep scores finite and comparable
    R = np.minimum(R, 1e6)

    # 5. Combine circumradius and direct edge length.
    #    Normalize R to the scale of distances so alpha is meaningful.
    #    Use a weighted sum: score = alpha * R + (1 - alpha) * d_curr
    alpha = 0.5
    scores = alpha * R + (1.0 - alpha) * d_curr

    # 6. Select the candidate with the minimum score
    best_idx = np.argmin(scores)

    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
