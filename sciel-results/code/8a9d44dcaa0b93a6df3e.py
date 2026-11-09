# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return np.sum(total == 0)


def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    import math
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for 54 points and 27 lines.
    
    Uses a hybrid construction combining vertical, horizontal, and diagonal lines
    on a structured grid to maximize incidences.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    # Start with the baseline grid construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate candidate lines
    candidate_lines = []
    
    # Vertical lines: x = c -> (1, 0, -c)
    for c in xs:
        candidate_lines.append((1, 0, -c))
    
    # Horizontal lines: y = c -> (0, 1, -c)
    for c in ys:
        candidate_lines.append((0, 1, -c))
    
    # Diagonal lines: y = a*x + b -> (a, -1, b)
    for a in range(-2, 3):
        if a == 0:
            continue  # Already have horizontal
        for b in range(-20, 20):
            candidate_lines.append((a, -1, b))
    
    # More diverse lines: y = (p/q)*x + b, represented as (p, -q, q*b)
    for p in range(-5, 6):
        for q in range(1, 6):
            if p == 0:
                continue
            if np.gcd(abs(p), q) != 1:
                continue
            for b in range(-20, 20):
                candidate_lines.append((p, -q, q * b))
    
    # Remove duplicates and invalid lines
    seen = set()
    unique_lines = []
    for line in candidate_lines:
        A, B, C = line
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
    
    # Compute incidences for each candidate line
    line_incidents = []
    for line in unique_lines:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        if count > 0:
            line_incidents.append((count, line))
    
    # Sort by incidence count (descending)
    line_incidents.sort(key=lambda x: -x[0])
    
    # Greedily select lines to maximize total incidences
    selected_lines = []
    total_incidents = 0
    used_line_set = set()
    
    for count, line in line_incidents:
        if len(selected_lines) >= 27:
            break
        if line in used_line_set:
            continue
        selected_lines.append(line)
        used_line_set.add(line)
        total_incidents += count
    
    # If we don't have 27 lines, fill with zero-incidence lines
    while len(selected_lines) < 27:
        for line in unique_lines:
            if line not in used_line_set:
                selected_lines.append(line)
                used_line_set.add(line)
                break
    
    lines = np.array(selected_lines[:27], dtype=np.int64)
    
    # Verify the result
    final_incidents = _incidence_count(points, lines)
    
    return points, lines


# EVOLVE-BLOCK-END
