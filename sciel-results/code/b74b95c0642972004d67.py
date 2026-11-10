import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580.0

    n_points = 54
    n_lines = 27

    # Construct 6x9 grid
    k = 6
    m = 9
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    # Generate all unique lines through pairs of grid points
    all_lines = {}  # normalized line -> (A, B, C)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in all_lines:
                all_lines[norm] = (A, B, C)

    # Count incidences for each line
    line_incidents = {}
    for norm, (A, B, C) in all_lines.items():
        count = 0
        for idx in range(n_points):
            x = int(points[idx, 0])
            y = int(points[idx, 1])
            if A * x + B * y + C == 0:
                count += 1
        line_incidents[norm] = count

    # Fixed lines: 9 horizontal + 6 vertical
    fixed_lines = []
    for j in range(m):
        fixed_lines.append((0, 1, -j))
    for i in range(k):
        fixed_lines.append((1, 0, -i))

    fixed_norms = set()
    for A, B, C in fixed_lines:
        norm = _normalize_line(A, B, C)
        fixed_norms.add(norm)

    # Candidate lines: all non-fixed lines, sorted by incidence count
    candidates = []
    for norm, (A, B, C) in all_lines.items():
        if norm not in fixed_norms:
            candidates.append((line_incidents[norm], A, B, C, norm))

    candidates.sort(key=lambda x: -x[0])

    # Greedy selection: pick top 12 unique lines
    selected = []
    selected_norms = set()
    for count, A, B, C, norm in candidates:
        if norm in selected_norms:
            continue
        if len(selected) >= 12:
            break
        selected.append((A, B, C))
        selected_norms.add(norm)

    lines_list = fixed_lines + selected

    # Ensure exactly 27 lines
    if len(lines_list) < n_lines:
        seen = set()
        for A, B, C in lines_list:
            seen.add(_normalize_line(A, B, C))
        while len(lines_list) < n_lines:
            a = np.random.randint(0, n_points)
            b = np.random.randint(0, n_points)
            while b == a:
                b = np.random.randint(0, n_points)
            x_a, y_a = int(points[a, 0]), int(points[a, 1])
            x_b, y_b = int(points[b, 0]), int(points[b, 1])
            A = y_b - y_a
            B = x_a - x_b
            C = -(A * x_a + B * y_a)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in seen:
                seen.add(norm)
                lines_list.append((A, B, C))

    final_lines = np.array(lines_list[:n_lines], dtype=np.int64)

    return points, final_lines
