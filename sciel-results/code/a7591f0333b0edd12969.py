"""
Simulated‑annealing coordinate search for BBOB f18 (Schaffers F7 ill‑conditioned).

Design family: deterministic, single‑pass, coordinate‑wise SA with geometric
cooling.  No random initialisation, no restarts, no NumPy vector operations per
step.  Every function call is a 1‑D perturbation of one coordinate of the
current best point, so the search is extremely cheap per call and can spend
the entire 500‑call budget on productive local moves.  The geometric schedule
starts with large steps (to cross the wide, shallow valleys of F7) and
progressively shrinks to fine‑grained exploitation, giving steady improvement
from the very first call (important for AOCC, which rewards early progress).
"""
import time
import math


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10 s safety margin before 330 s kill

        budget = self.budget
        dim = self.dim

        # Deterministic start at the centre of the box.  For the symmetric,
        # multi‑minima landscape of F7 this is a reasonable, cheap seed.
        x = [0.0] * dim
        fx = func(x)
        n_evals = 1

        if n_evals >= budget:
            return x, fx

        # Geometric cooling schedule: step size for coordinate i at call k.
        # s0 is chosen so that the first step is a large fraction of the box
        # (helps cross wide valleys); s_min is fine enough for final polish.
        s0 = 4.0
        s_min = 1e-5
        # Number of calls we will make after the initial evaluation.
        n_steps = budget - n_evals
        # Geometric ratio so that s0 * ratio**n_steps == s_min
        ratio = (s_min / s0) ** (1.0 / n_steps)

        # Deterministic coordinate cycling: visit coordinates in a fixed,
        # well‑mixed order (GCD(7, dim)=1 for dim=5 gives a full cycle).
        # No randomness -> reproducible, and no RNG overhead.
        coord_seq = [(7 * k) % dim for k in range(n_steps)]

        # Alternate direction to avoid bias; also helps escape flat regions.
        # Use a simple deterministic sign pattern.
        sign_seq = [1.0 if (k % 4 < 2) else -1.0 for k in range(n_steps)]

        best_x = x[:]
        best_fx = fx

        # Annealing "temperature" for accepting uphill moves.
        # We accept uphill moves with probability exp(-delta / T).
        # T cools geometrically from T0 to T_min.
        T0 = 1.0
        T_min = 1e-4
        T_ratio = (T_min / T0) ** (1.0 / n_steps)
        T = T0

        # Deterministic pseudo‑random for acceptance (xorshift on an integer).
        # Cheap and no import needed.
        seed = 123456789
        def xorshift():
            nonlocal seed
            seed ^= (seed << 13) & 0xFFFFFFFFFFFFFFFF
            seed ^= (seed >> 7)
            seed ^= (seed << 17) & 0xFFFFFFFFFFFFFFFF
            return (seed & 0xFFFFFFFFFFFFFFFF) / 0xFFFFFFFFFFFFFFFF

        for k in range(n_steps):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break

            i = coord_seq[k]
            direction = sign_seq[k]

            # Current step size for this coordinate
            step = s0 * (ratio ** k)
            step = max(step, s_min)

            # Candidate: perturb coordinate i by +/- step
            x_new = x[:]
            x_new[i] += direction * step
            # Clip to box
            if x_new[i] < -5.0:
                x_new[i] = -5.0
            elif x_new[i] > 5.0:
                x_new[i] = 5.0

            fx_new = func(x_new)
            n_evals += 1

            delta = fx_new - fx
            if delta <= 0:
                # Downhill or equal: always accept
                x = x_new
                fx = fx_new
                if fx < best_fx:
                    best_fx = fx
                    best_x = x[:]
            else:
                # Uphill: accept with probability exp(-delta / T)
                # Use a cheap approximation: if delta is small relative to T,
                # accept; otherwise reject.  We use the actual exponential
                # with a deterministic random number.
                if delta < T:  # cheap pre‑check
                    r = xorshift()
                    if r < math.exp(-delta / T):
                        x = x_new
                        fx = fx_new
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x[:]
                # else: reject

            # Cool temperature
            T *= T_ratio
            if T < T_min:
                T = T_min

        return best_x, best_fx
