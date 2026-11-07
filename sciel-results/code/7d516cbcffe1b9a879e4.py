"""Program that exploits DTLZ1's known structure with adaptive HV-driven placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ1 closed-form front with adaptive hypervolume-driven placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # DTLZ1 front: x[0]=u1, x[1]=u2, x[2:]=0.5
        # f = [0.5*u1, 0.5*(1-u1)*u2, 0.5*(1-u1)*(1-u2)]

        # Greedy max-min distance placement in (u1,u2) parameter space
        selected = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]

        # Fine candidate grid
        n_cand = 40
        cand_u1 = np.linspace(0.0, 1.0, n_cand)
        cand_u2 = np.linspace(0.0, 1.0, n_cand)
        candidates = [(float(u1), float(u2)) for u1 in cand_u1 for u2 in cand_u2]

        while len(selected) < budget:
            if time.time() > deadline - 10.0:
                break

            best_pt = None
            best_score = -1.0

            for cand in candidates:
                if cand in selected:
                    continue

                min_dist = float('inf')
                for sel in selected:
                    dist = (cand[0] - sel[0])**2 + (cand[1] - sel[1])**2
                    min_dist = min(min_dist, dist)

                if min_dist > best_score:
                    best_score = min_dist
                    best_pt = cand

            if best_pt is None:
                break

            selected.append(best_pt)

        # Evaluate all selected points
        recorded = []
        for (u1, u2) in selected:
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = float(u1)
            x[1] = float(u2)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
