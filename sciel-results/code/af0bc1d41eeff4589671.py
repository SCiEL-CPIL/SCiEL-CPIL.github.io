"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    # Try different aspect ratios for the rectangle
    # w + h = 2, ar = h/w, so w = 2/(1+ar), h = 2 - w
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5, 1.1, 1.4, 1.8, 0.7, 2.2]

    for ar in aspect_ratios:
        if time.time() > deadline - 20:  # Leave 20s margin
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        # Generate a good initial layout using hexagonal-ish packing
        # For a grid, use a staggered pattern
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))

        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = []
        ys = []
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            # Stagger odd rows
            if row % 2 == 1:
                x += 0.25 * w / n_cols
                # Keep within bounds
                if x > w - w/n_cols * 0.25:
                    x -= 0.5 * w / n_cols
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)

        # Compute initial radius
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])

        r0 = 0.35 * min_dist
        radii = np.full(N, r0)

        def unpack(vars_):
            x = vars_[0::3]
            y = vars_[1::3]
            r = vars_[2::3]
            return x, y, r

        def objective(vars_):
            x, y, r = unpack(vars_)
            return -np.sum(r)

        def constraints(vars_):
            x, y, r = unpack(vars_)
            cons = []
            # Circles inside rectangle
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, r=r: x[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, w=w, r=r: w - x[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, r=r: y[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, h=h, r=r: h - y[i] - r[i]})
            # No overlap
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({'type': 'ineq', 'fun': lambda i=i, j=j, x=x, y=y, r=r:
                                 np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) - r[i] - r[j]})
            # Positive radii
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda i=i, r=r: r[i]})
            return cons

        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = radii[i]

        try:
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-7}
            )

            if result.success or result.fun < 0:
                x, y, r = unpack(result.x)
                score = np.sum(r)

                # Verify validity
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
                        valid = False
                        break
                    if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                        valid = False
                        break
                    if r[i] <= 0:
                        valid = False
                        break

                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break

                if valid and score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])

        except Exception:
            continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
