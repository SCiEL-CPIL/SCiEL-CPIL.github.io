"""Hexagonal packing in variable-aspect rectangle, perimeter 4, 21 circles."""
import numpy as np
from itertools import combinations_with_replacement, permutations

N = 21

def _hex_positions(row_counts):
    """Generate center positions (x, y) with r=1, horizontal spacing 2, vertical spacing sqrt(3)."""
    pts = []
    y = 0.0
    for idx, m in enumerate(row_counts):
        offset = (idx % 2) * 1.0  # offset by 1 (= r) for alternating rows
        for i in range(m):
            x = i * 2.0 + offset
            pts.append((x, y))
        y += np.sqrt(3.0)
    return np.array(pts)

def _best_r_for_pattern(row_counts):
    """Given row pattern, compute max r such that perimeter <= 4, return 21*r."""
    pts = _hex_positions(row_counts)
    # With radius r, center spacing scales: actual x = r * pts_x, y = r * pts_y
    # Bounding box: [r*(min_x - 1), r*(max_x + 1)] x [r*(min_y - 1), r*(max_y + 1)]
    # since with r=1, the circle extends 1 unit beyond center.
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    # Width = r * (max_x - min_x + 2), Height = r * (max_y - min_y + 2)
    W0 = max_x - min_x + 2.0
    H0 = max_y - min_y + 2.0
    # Perimeter = 2 * r * (W0 + H0) <= 4  =>  r <= 2 / (W0 + H0)
    r = 2.0 / (W0 + H0)
    # Verify no overlap: min distance between centers should be >= 2r.
    # In our construction, min center distance is 2 (horizontal) or sqrt(3) (diagonal) in r=1 units.
    # sqrt(3) ~ 1.732 < 2, so diagonal neighbors are closer! Need to check.
    # Actually in hex packing, nearest neighbors are at distance 2 (horizontal) and also at distance 2 (diagonal, since vertical is sqrt(3) and horizontal offset 1, so distance = sqrt(1^2 + 3) = 2). Good, all nearest = 2.
    # So min distance = 2 in r=1 units => 2r actual, which equals 2r. So they touch. Good.
    return 21.0 * r, r

def construct():
    # Try many row patterns summing to 21
    best_val = 0.0
    best_pattern = None
    # Generate patterns: list of positive ints summing to 21, length 2..8
    def gen_patterns(total, max_len):
        results = []
        def helper(remaining, length, current):
            if length == 0:
                if remaining == 0:
                    results.append(tuple(current))
                return
            if length < 0 or remaining < length:
                return
            for v in range(1, remaining - length + 2):
                current.append(v)
                helper(remaining - v, length - 1, current)
                current.pop()
        for L in range(2, max_len + 1):
            helper(total, L, [])
        return results
    
    patterns = gen_patterns(N, 8)
    for pat in patterns:
        val, r = _best_r_for_pattern(pat)
        if val > best_val:
            best_val = val
            best_pattern = pat
    
    # Build the actual circles
    pts = _hex_positions(best_pattern)
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    W0 = max_x - min_x + 2.0
    H0 = max_y - min_y + 2.0
    r = 2.0 / (W0 + H0)
    
    # Scale and center
    xs = r * pts[:, 0]
    ys = r * pts[:, 1]
    # Center the bounding box at origin
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    xs -= cx
    ys -= cy
    
    circles = np.column_stack([xs, ys, np.full(N, r)])
    return circles
