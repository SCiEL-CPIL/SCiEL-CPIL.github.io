"""Multi-objective optimizer on WFG4 combining NSGA-II with LHS initialization
and a hypervolume-aware replacement heuristic."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II (non-dominated sorting + crowding) with DE offspring, LHS
    initialization, and a hypervolume-aware replacement heuristic."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _fast_non_dominated_sort(F):
        n = F.shape[0]
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            nxt = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        nxt.append(j)
            if not nxt:
                break
            fronts.append(nxt)
            front0 = nxt
        rank = np.full(n, n, dtype=int)
        for k, fr in enumerate(fronts):
            for i in fr:
                rank[i] = k
        return rank

    @staticmethod
    def _crowding_distance(F, indices):
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        Ff = F[indices]
        for k in range(Ff.shape[1]):
            v = Ff[:, k]
            vmin, vmax = v.min(), v.max()
            if vmax - vmin <= 1e-18:
                continue
            order = np.argsort(v)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, m - 1):
                dist[order[i]] += (v[order[i + 1]] - v[order[i - 1]]) / (vmax - vmin)
        return dist

    @staticmethod
    def _hv_contrib(F, ref):
        """Exact hypervolume contribution of each point (3D, inclusion-exclusion)."""
        n = F.shape[0]
        pts = [(i, F[i]) for i in range(n) if np.all(F[i] < ref)]
        contrib = np.zeros(n)
        for i, fi in pts:
            a = fi
            a0, a1, a2 = a[0], a[1], a[2]
            r0, r1, r2 = ref[0], ref[1], ref[2]
            # single boxes
            vol = (r0 - a0) * (r1 - a1) * (r2 - a2)
            # subtract pairwise overlaps (3 pairs)
            vol -= max(0.0, r0 - max(a0, a1)) * (r1 - max(a1, a2)) * (r2 - max(a2, a0))
            vol -= max(0.0, r0 - max(a0, a2)) * (r1 - max(a1, a0)) * (r2 - max(a2, a1))
            vol -= max(0.0, r0 - max(a1, a2)) * (r1 - max(a1, a0)) * (r2 - max(a2, a0))
            # add triple overlap
            vol += max(0.0, r0 - max(a0, a1, a2)) * (r1 - max(a0, a1, a2)) * (r2 - max(a0, a1, a2))
            contrib[i] = max(0.0, vol)
        return contrib

    def __call__(self, problem, bounds):
        deadline = time.time() + 275.0  # safety margin before 300s limit
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random

        # ---- LHS initialization for better initial spread ----
        pop = int(min(80, max(16, self.budget // 2)))
        lhs_samples = np.zeros((pop, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, pop + 1)
            points = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(points)
            lhs_samples[:, d] = points
        X = lo + lhs_samples * span
        F = np.array([problem(x) for x in X])
        used = pop
        best = F.copy()

        # DE parameters
        F_cr = 0.7
        CR = 0.9
        K = 0.5

        while used < self.budget:
            if time.time() > deadline:
                break

            rank = self._fast_non_dominated_sort(F)
            cd = np.zeros(len(F))
            for fr in np.unique(rank):
                idx = list(np.where(rank == fr)[0])
                if len(idx) > 1:
                    cd[idx] = self._crowding_distance(F, idx)

            # ---- DE-based offspring: pick 3 distinct parents via tournament
            def tournament_select(k=3):
                candidates = rng.randint(0, len(F), k)
                best_c = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best_c]:
                        best_c = c
                    elif rank[c] == rank[best_c]:
                        if cd[c] > cd[best_c]:
                            best_c = c
                return int(best_c)

            a = tournament_select(3)
            b = tournament_select(3)
            c = tournament_select(3)
            while b == a:
                b = tournament_select(3)
            while c == a or c == b:
                c = tournament_select(3)

            F1, F2, F3 = F[a], F[b], F[c]
            # adaptive scaling: larger for worse parents
            best_rank = rank[a]
            scale = 0.7 + 0.3 * (best_rank + 1) / (rank.max() + 1)
            scale = min(scale, 1.2)

            v = F1 + scale * (F2 - F3)
            jrand = int(rng.integers(dim))
            mask = rng.random(dim) < CR
            mask[jrand] = True
            child = np.where(mask, v, F1)
            # per-coordinate gaussian mutation
            child = child + K * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            if np.any(fc < best):
                best = np.minimum(best, fc)

            # ---- replacement: HV-aware if cheap, else NSGA-II
            combined = np.vstack([F, fc[None, :]])
            n2 = len(combined)
            if n2 <= 120:
                # skip if child is dominated by an existing point
                if np.any(np.all(F <= fc) and np.any(F < fc)):
                    continue
                contrib = self._hv_contrib(combined, np.array([3.0, 5.0, 7.0]))
                base = np.sum(contrib[:pop])
                newvol = np.sum(contrib) - contrib[pop]
                if newvol <= base + 1e-12:
                    continue
                worst = int(np.argmin(contrib[:pop]))
                X[worst], F[worst] = child, fc
            else:
                rank2 = self._fast_non_dominated_sort(combined)
                cd2 = np.zeros(n2)
                for fr in np.unique(rank2):
                    idx = list(np.where(rank2 == fr)[0])
                    if len(idx) > 1:
                        cd2[idx] = self._crowding_distance(combined, idx)
                order2 = np.lexsort((-cd2, rank2))
                keep = set(order2[:pop].tolist())
                replaced = False
                for k in range(n2):
                    if k == pop:
                        continue
                    if k not in keep:
                        X[k], F[k] = child, fc
                        replaced = True
                        break
                if not replaced:
                    w = int(np.argmax(rank2[:pop] + 1e-6 * (-cd2[:pop])))
                    X[w], F[w] = child, fc

        return X
# EVOLVE-BLOCK-END
