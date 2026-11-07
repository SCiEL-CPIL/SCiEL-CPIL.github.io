"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + adaptive multi-start local hill climb."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        def check_deadline():
            return calls >= budget or time.time() > deadline

        # Phase 1: Deterministic grid search - more points for better initial coverage
        # Vertices: all combinations of -5 and 5
        vertices = []
        for mask in range(2 ** n):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            vertices.append(v)
        # Add center
        vertices.append(np.zeros(n))
        # Add axis points: +/-2, +/-4 along each axis
        for i in range(n):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(n)
                p[i] = val
                vertices.append(p)
        # Add face centers: points with two non-zero coords
        for i in range(n):
            for j in range(i + 1, n):
                for vi in [-3.0, 3.0]:
                    for vj in [-3.0, 3.0]:
                        p = np.zeros(n)
                        p[i] = vi
                        p[j] = vj
                        vertices.append(p)

        # Evaluate all grid points, track top few
        results = []
        for v in vertices:
            if check_deadline():
                break
            f = safe_call(v)
            if f is not None:
                results.append((f, v.copy()))

        if not results:
            # Fallback: evaluate at origin
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            results = [(best_f, best_x)]

        # Sort by function value, keep top 5 for multi-start
        results.sort(key=lambda x: x[0])
        best_x = results[0][1]
        best_f = results[0][0]
        top_starts = [r[1] for r in results[:min(5, len(results))]]

        # Phase 2: Adaptive local hill climb from best point first
        # Use a smarter step schedule with early termination per dimension
        step_sizes = [3.0, 1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375]
        
        def local_hill_climb(start_x, start_f, max_steps_per_dim=3):
            nonlocal best_x, best_f, calls
            cx = start_x.copy()
            cf = start_f
            for step in step_sizes:
                if check_deadline():
                    break
                improved_any = False
                for i in range(n):
                    if check_deadline():
                        break
                    for direction in [-1, 1]:
                        if check_deadline():
                            break
                        y = cx.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < cf:
                            cx = y
                            cf = fy
                            improved_any = True
                            if cf < best_f:
                                best_x = cx.copy()
                                best_f = cf
                if not improved_any:
                    break
            return cx, cf

        # Refine from the best start point with more budget
        local_hill_climb(top_starts[0], results[0][0])

        # Refine from other top starts with less budget each
        for start_idx in range(1, len(top_starts)):
            if check_deadline():
                break
            # Use fewer step sizes for secondary starts
            for step in step_sizes[:4]:
                if check_deadline():
                    break
                improved_any = False
                sx = top_starts[start_idx].copy()
                sf = results[start_idx][0]
                for i in range(n):
                    if check_deadline():
                        break
                    for direction in [-1, 1]:
                        if check_deadline():
                            break
                        y = sx.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < sf:
                            sx = y
                            sf = fy
                            improved_any = True
                            if sf < best_f:
                                best_x = sx.copy()
                                best_f = sf
                if not improved_any:
                    break

        # Phase 3: Final fine-tuning pass from the best point
        if not check_deadline():
            fine_steps = [0.01, 0.005, 0.002, 0.001]
            cx = best_x.copy()
            cf = best_f
            for step in fine_steps:
                if check_deadline():
                    break
                improved_any = False
                for i in range(n):
                    if check_deadline():
                        break
                    for direction in [-1, 1]:
                        if check_deadline():
                            break
                        y = cx.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < cf:
                            cx = y
                            cf = fy
                            improved_any = True
                            if cf < best_f:
                                best_x = cx.copy()
                                best_f = cf
                if not improved_any:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
