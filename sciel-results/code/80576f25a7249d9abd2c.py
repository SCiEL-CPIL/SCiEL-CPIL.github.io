# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    max_time = 590.0

    def calc_penalty_vec(cx, cy, s, tol=1e-7):
        """Vectorized penalty: boundary + overlap area. Lower is better."""
        x_min = cx - s / 2.0
        x_max = cx + s / 2.0
        y_min = cy - s / 2.0
        y_max = cy + s / 2.0

        pen = float(np.sum(np.maximum(0, -x_min)) +
                    np.sum(np.maximum(0, x_max - 1)) +
                    np.sum(np.maximum(0, -y_min)) +
                    np.sum(np.maximum(0, y_max - 1)) +
                    np.sum(np.maximum(0, -s)))

        # Overlap: for each pair, if both axes overlap, add area
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                half = (s[i] + s[j]) / 2.0
                ox = half - abs(dx)
                oy = half - abs(dy)
                if ox > 0 and oy > 0:
                    pen += ox * oy
        return pen

    def is_valid(cx, cy, s, tol=1e-7):
        if np.any(s <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(cx)) or np.any(~np.isfinite(cy)):
            return False
        x_min = cx - s / 2.0
        x_max = cx + s / 2.0
        y_min = cy - s / 2.0
        y_max = cy + s / 2.0
        if np.any(x_min < -tol) or np.any(x_max > 1 + tol):
            return False
        if np.any(y_min < -tol) or np.any(y_max > 1 + tol):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def init_2x5():
        """2 rows x 5 cols of 0.2 squares. Sum = 2.0"""
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            row = i // 5
            col = i % 5
            cx[i] = col * 0.2 + 0.1
            cy[i] = row * 0.2 + 0.1
            s[i] = 0.2
        return cx, cy, s

    def init_3x3_plus():
        """3x3 grid of 1/3 squares + 1 small. Sum ~ 3.0 + small"""
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        for i in range(9):
            row = i // 3
            col = i % 3
            cx[i] = col / 3.0 + 1.0 / 6.0
            cy[i] = row / 3.0 + 1.0 / 6.0
            s[i] = 1.0 / 3.0
        # 10th: small square in a corner gap - but there are no gaps
        # Shrink the 9 slightly and add a 10th
        s[:9] = 0.32
        # Recompute positions to be valid
        cx[:9] = np.array([0.16, 0.5, 0.84, 0.16, 0.5, 0.84, 0.16, 0.5, 0.84])
        cy[:9] = np.array([0.16, 0.16, 0.16, 0.5, 0.5, 0.5, 0.84, 0.84, 0.84])
        # 10th square small in center gap
        cx[9] = 0.5
        cy[9] = 0.5
        s[9] = 0.0
        # This won't be valid; let SA fix it
        return cx, cy, s

    def init_shelf_hetero():
        """Heterogeneous shelf: larger squares on bottom, smaller on top.
        Bottom row: 2 large (0.45 each), top row: 8 smaller.
        Actually let's do: bottom 3 of size 0.33, top 7 of size ~0.14
        Sum = 3*0.33 + 7*0.14 = 0.99 + 0.98 = 1.97
        Better: 2 large (0.48) + 8 small (0.125) = 0.96 + 1.0 = 1.96
        Let's try: 4 large (0.25) in 2x2 bottom-left, 6 small (0.166) in 2x3 top-right
        This is getting complex. Let's use a known good pattern.
        """
        # Pattern: 5 in bottom row (0.2), 5 in top row (0.2) = 2.0
        # Or: 2 big (0.4) side by side bottom, 8 small (0.125) in 4x2 top
        # 2*0.4 + 8*0.125 = 0.8 + 1.0 = 1.8
        # Let's try: 1 big (0.5) in corner, 9 small (0.111) in remaining space
        # 0.5 + 9*0.111 = 0.5 + 0.999 = 1.499
        # Best simple: 2x5 grid = 2.0
        # Try heterogeneous: bottom 5 of 0.22, top 5 of 0.16
        # 5*0.22 + 5*0.16 = 1.1 + 0.8 = 1.9
        # But 5 of 0.22 in a row needs 1.1 > 1.0, won't fit.
        # Bottom 4 of 0.24 (needs 0.96), top 6 of 0.16 (needs 0.96)
        # 4*0.24 + 6*0.16 = 0.96 + 0.96 = 1.92
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        # Bottom row: 4 squares of 0.24
        for i in range(4):
            cx[i] = i * 0.24 + 0.12
            cy[i] = 0.12
            s[i] = 0.24
        # Top row: 6 squares of 0.16
        for i in range(6):
            cx[4 + i] = i * 0.16 + 0.08
            cy[4 + i] = 0.12 + 0.24 + 0.08  # = 0.44
            s[4 + i] = 0.16
        return cx, cy, s

    def init_shelf_3row():
        """3 rows: bottom 4 of 0.25, middle 3 of 0.2, top 3 of 0.15
        Sum = 4*0.25 + 3*0.2 + 3*0.15 = 1.0 + 0.6 + 0.45 = 2.05
        Check: bottom 4*0.25=1.0 fits. Middle 3*0.2=0.6 fits. Top 3*0.15=0.45 fits.
        Heights: 0.25 + 0.2 + 0.15 = 0.6 < 1.0, fits with room.
        """
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        idx = 0
        # Bottom row: 4 of 0.25
        for i in range(4):
            cx[idx] = i * 0.25 + 0.125
            cy[idx] = 0.125
            s[idx] = 0.25
            idx += 1
        # Middle row: 3 of 0.2
        for i in range(3):
            cx[idx] = i * 0.2 + 0.1
            cy[idx] = 0.25 + 0.1 + 0.05  # gap
            s[idx] = 0.2
            idx += 1
        # Top row: 3 of 0.15
        for i in range(3):
            cx[idx] = i * 0.15 + 0.075
            cy[idx] = 0.25 + 0.2 + 0.15 / 2 + 0.05  # approximate
            s[idx] = 0.15
            idx += 1
        return cx, cy, s

    def init_random_valid():
        """Random valid packing by placing squares one at a time."""
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        # Start with a base size and try to place
        base = 0.15
        for i in range(n):
            placed = False
            for attempt in range(200):
                si = base + random.uniform(-0.05, 0.05)
                si = max(0.05, min(si, 0.4))
                xi = random.uniform(si / 2, 1 - si / 2)
                yi = random.uniform(si / 2, 1 - si / 2)
                # Check against already placed
                ok = True
                for j in range(i):
                    t = (si + s[j]) / 2 - 1e-7
                    if abs(xi - cx[j]) < t and abs(yi - cy[j]) < t:
                        ok = False
                        break
                if ok:
                    cx[i] = xi
                    cy[i] = yi
                    s[i] = si
                    placed = True
                    break
            if not placed:
                # Fallback: small square in corner
                si = 0.05
                cx[i] = si / 2 + i * 0.01
                cy[i] = si / 2 + i * 0.01
                s[i] = si
        return cx, cy, s

    # Generate multiple initial configurations
    inits = [
        init_2x5(),
        init_shelf_hetero(),
        init_shelf_3row(),
        init_random_valid(),
    ]

    best_cx, best_cy, best_s = None, None, None
    best_score = 0.0

    for cx0, cy0, s0 in inits:
        if not is_valid(cx0, cy0, s0):
            # Try to fix by shrinking
            s0 = s0 * 0.9
            if not is_valid(cx0, cy0, s0):
                continue
        score0 = float(np.sum(s0))
        if score0 > best_score:
            best_score = score0
            best_cx, best_cy, best_s = cx0.copy(), cy0.copy(), s0.copy()

    # If no valid init, use 2x5
    if best_cx is None:
        cx0, cy0, s0 = init_2x5()
        best_cx, best_cy, best_s = cx0.copy(), cy0.copy(), s0.copy()
        best_score = float(np.sum(s0))

    # SA optimization from best init
    cx, cy, s = best_cx.copy(), best_cy.copy(), best_s.copy()
    curr_pen = calc_penalty_vec(cx, cy, s)
    curr_score = float(np.sum(s)) - curr_pen

    T = 0.05
    T_min = 1e-6
    cool = 0.9995

    deadline = start_time + max_time - 5.0

    while time.time() < deadline:
        # Batch of perturbations for speed
        cx_new = cx.copy()
        cy_new = cy.copy()
        s_new = s.copy()

        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(n, num_perturb, replace=False)

        scale = 0.02 * (1.0 + T)
        for idx in indices:
            cx_new[idx] += np.random.normal(0, scale)
            cy_new[idx] += np.random.normal(0, scale)
            s_new[idx] += np.random.normal(0, scale * 0.8)
            s_new[idx] = max(0.01, min(s_new[idx], 0.9))
            cx_new[idx] = max(-0.3, min(cx_new[idx], 1.3))
            cy_new[idx] = max(-0.3, min(cy_new[idx], 1.3))

        pen_new = calc_penalty_vec(cx_new, cy_new, s_new)
        score_new = float(np.sum(s_new)) - pen_new
        delta = score_new - curr_score

        if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-10)):
            cx, cy, s = cx_new, cy_new, s_new
            curr_pen = pen_new
            curr_score = score_new

            if pen_new < 1e-8:
                sc = float(np.sum(s))
                if sc > best_score:
                    best_score = sc
                    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()

        T = max(T * cool, T_min)

    # Final validation
    if not is_valid(best_cx, best_cy, best_s):
        # Fallback to 2x5 grid
        cx0, cy0, s0 = init_2x5()
        best_cx, best_cy, best_s = cx0, cy0, s0

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
