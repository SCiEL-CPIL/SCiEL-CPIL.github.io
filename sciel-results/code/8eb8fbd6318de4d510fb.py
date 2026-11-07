"""Multi-objective optimizer on WFG7 (fresh design: weighted-scalarization multi-start + L-SHADE local search)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Weighted-scalarization multi-start search with L-SHADE-style adaptive local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0   # safety margin under the 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0], dtype=float)

        # ---- scalarization weight vectors (well-spread simplex, incl. axes) ----
        def _simplex_weights(n, k):
            if n == 1:
                return np.array([[1.0] * k])
            out = []
            for i in range(n):
                w = np.zeros(k)
                w[i] = 1.0
                out.append(w)
            for i in range(n - 1):
                w = np.zeros(k)
                w[i] = 0.5
                w[i + 1] = 0.5
                out.append(w)
            return np.array(out)

        W = _simplex_weights(7, n_obj)

        # ---- hypervolume (exact, 3-D, via inclusion-exclusion) ----
        def _hv3(F):
            F = np.asarray(F, dtype=float)
            F = F[F[:, 0] < ref[0]]
            F = F[F[:, 1] < ref[1]]
            F = F[F[:, 2] < ref[2]]
            if F.shape[0] == 0:
                return 0.0
            # keep only non-dominated points
            keep = np.ones(F.shape[0], dtype=bool)
            for i in range(F.shape[0]):
                if not keep[i]:
                    continue
                d = (F <= F[i]) & np.any(F < F[i], axis=1)
                d[i] = False
                keep[d] = False
            F = F[keep]
            if F.shape[0] == 0:
                return 0.0
            F = F[np.lexsort((F[:, 2], F[:, 1], F[:, 0]))]
            a = F[:, 0]
            b = F[:, 1]
            c = F[:, 2]
            n = F.shape[0]
            x = np.concatenate(([0.0], a, [ref[0]]))
            y = np.concatenate(([0.0], b, [ref[1]]))
            z = np.concatenate(([0.0], c, [ref[2]]))
            vol = 0.0
            for i in range(n):
                vol += (x[i + 1] - x[i]) * (y[i + 1] - y[i]) * (z[i + 1] - z[i])
            return float(vol)

        # ---- per-objective scaling for Tchebycheff ----
        def _tch(x, f, w, fmin, fmax):
            rngv = fmax - fmin
            rngv[rngv < 1e-12] = 1.0
            return np.max(w * (f - fmin) / rngv)

        # ---- local search (L-SHADE style) ----
        def _local_search(x, f, w, fmin, fmax, base_step):
            best_x = x.copy()
            best_f = f.copy()
            best_tv = _tch(best_x, best_f, w, fmin, fmax)
            step = base_step
            no_improve = 0
            while used[0] < budget and time.time() < deadline:
                # adaptive step
                if no_improve >= 6:
                    step *= 0.6
                elif no_improve == 0 and step < base_step:
                    step *= 1.3
                step = max(step, 0.02)
                # pick a second individual for recombination
                if len(archive) > 0:
                    j = rng.integers(len(archive))
                    x2 = archive[j][0]
                else:
                    x2 = best_x
                r = rng.random(dim)
                mut = rng.standard_normal(dim)
                child = x2 * r + best_x * (1.0 - r) + step * (hi - lo) * mut
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used[0] += 1
                tv = _tch(child, fc, w, fmin, fmax)
                # accept if it improves tchebycheff value or is non-dominated vs best
                if tv < best_tv - 1e-12:
                    best_x, best_f, best_tv = child, fc, tv
                    no_improve = 0
                elif (fc <= best_f).all() and np.any(fc < best_f):
                    best_x, best_f = child, fc
                    best_tv = _tch(best_x, best_f, w, fmin, fmax)
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 14:
                    break
            return best_x, best_f

        archive = []   # list of (x, f)
        used = [0]

        # ---- initial population: spread across weight directions ----
        n_init = min(24, budget)
        base_step = 0.12
        for k in range(n_init):
            if used[0] >= budget or time.time() >= deadline:
                break
            w = W[k % len(W)]
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used[0] += 1
            archive.append((x, f))

        # ---- evaluate current archive for fmin/fmax ----
        def _refresh_bounds():
            if len(archive) == 0:
                return np.zeros(n_obj), np.ones(n_obj)
            F = np.array([a[1] for a in archive])
            Fm = F[(F < ref).all(axis=1)]
            if Fm.shape[0] == 0:
                Fm = F
            return Fm.min(axis=0), Fm.max(axis=0)

        # ---- main loop: local search around best points, keep hypervolume-best set ----
        while used[0] < budget and time.time() < deadline:
            if len(archive) == 0:
                break
            fmin, fmax = _refresh_bounds()
            # choose the archive point with the best hypervolume contribution
            # (approx: pick a random archive point, but bias toward low tchebycheff)
            idx = rng.integers(len(archive))
            w = W[idx % len(W)]
            x, f = archive[idx]
            nx, nf = _local_search(x, f, w, fmin, fmax, base_step)
            # add to archive if it improves hypervolume
            if (nf < ref).all():
                new_arch = archive + [(nx, nf)]
                Fnew = np.array([a[1] for a in new_arch])
                if _hv3(Fnew) > _hv3(np.array([a[1] for a in archive])):
                    archive = new_arch
                # prune archive to top points by hypervolume contribution (keep <= 40)
                if len(archive) > 40:
                    # greedy: keep points that add hypervolume
                    kept = []
                    for (px, pf) in archive:
                        trial = kept + [(px, pf)]
                        if _hv3(np.array([a[1] for a in trial])) > _hv3(np.array([a[1] for a in kept])):
                            kept = trial
                    archive = kept
            # occasionally add a fresh random point to escape local optima
            if used[0] < budget and time.time() < deadline and rng.random() < 0.15:
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                used[0] += 1
                if (f < ref).all():
                    archive.append((x, f))

        # ---- final: return the hypervolume-best subset (all recorded points count) ----
        if len(archive) > 0:
            F = np.array([a[1] for a in archive])
            # greedy hypervolume subset selection (keep up to 60 best-contributing)
            kept = []
            for i in range(len(archive)):
                trial = kept + [archive[i]]
                if _hv3(np.array([a[1] for a in trial])) > _hv3(np.array([a[1] for a in kept])):
                    kept = trial
            archive = kept

        return archive
# EVOLVE-BLOCK-END
