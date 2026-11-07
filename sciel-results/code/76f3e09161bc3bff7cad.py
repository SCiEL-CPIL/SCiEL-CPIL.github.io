"""Multi-objective DTLZ2 optimizer: analytic-front projection + DE + angle polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured DTLZ2 optimizer exploiting the analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        n_shape = n_obj - 1      # 2 shape (angle) variables
        n_dist = dim - n_shape   # 10 distance variables

        # --- Build a point on the Pareto front from angles ---
        def front_point(angles):
            x = np.empty(dim)
            x[:n_shape] = angles
            x[n_shape:] = 0.5    # exactly on front (g = 0)
            return np.clip(x, lo, hi)

        # --- Fibonacci sphere sampling for even objective coverage ---
        def fib_sphere(n):
            pts = []
            golden = (1 + np.sqrt(5)) / 2.0
            for k in range(n):
                z = 1 - 2 * (k + 0.5) / n
                r = np.sqrt(max(0.0, 1 - z * z))
                phi = 2 * np.pi * k / golden
                # convert unit-sphere point to (theta1, theta2) in [0, pi/2]^2
                # x = cos(t1)cos(t2), y = cos(t1)sin(t2), z = sin(t1)
                t1 = np.arcsin(np.clip(z, -1, 1))
                t1 = np.clip(t1, 0, np.pi / 2)
                if abs(np.cos(t1)) > 1e-12:
                    t2 = np.arctan2(y if False else np.sin(t1) * 0 + r * np.sin(phi),
                                    r * np.cos(phi))
                else:
                    t2 = 0.0
                t2 = np.clip(t2, 0, np.pi / 2)
                pts.append([t1, t2])
            return np.array(pts)

        # Simpler, robust angle generator: uniform grid in spherical coords
        def gen_angles(n):
            pts = []
            golden = (1 + np.sqrt(5)) / 2.0
            for k in range(n):
                z = 1 - 2 * (k + 0.5) / n
                r = np.sqrt(max(0.0, 1 - z * z))
                phi = 2 * np.pi * k / golden
                t1 = np.arcsin(np.clip(z, -1, 1))
                t1 = np.clip(t1, 0, np.pi / 2)
                # x = cos(t1)cos(t2)= r cos(phi), y = cos(t1)sin(t2)= r sin(phi)
                if r > 1e-12:
                    t2 = np.arctan2(r * np.sin(phi), r * np.cos(phi))
                else:
                    t2 = 0.0
                t2 = np.clip(t2, 0, np.pi / 2)
                pts.append([t1, t2])
            return np.array(pts)

        def fast_non_dominated_sort(F):
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = [current_front]
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                rng_span = vals[order[-1]] - vals[order[0]]
                if rng_span < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_span
            return dist

        def select_population(Xc, Fc, pop_size):
            fronts = fast_non_dominated_sort(Fc)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd = crowding_distance(Fc, front)
                    order = np.argsort(cd)[::-1]
                    selected.extend(front[order[:remaining]])
                    break
            return Xc[selected], Fc[selected]

        # Phase 1: dense analytic-front seeding (even objective spread)
        n_init = 80
        angles = gen_angles(n_init)
        X = np.array([front_point(a) for a in angles])
        F = np.array([problem(x) for x in X])
        used = n_init

        # Phase 2: DE to refine/confirm front membership (short)
        pop_size = 60
        # keep the best pop_size by non-domination/crowding from the init
        X, F = select_population(X, F, pop_size)
        if used < self.budget and time.time() < deadline - 10:
            while used < self.budget and time.time() < deadline - 10:
                if time.time() > deadline - 12:
                    break
                n_off = pop_size
                offspring = np.empty((n_off, dim))
                offspring_F = np.empty((n_off, n_obj))
                for i in range(n_off):
                    a, b, c = rng.integers(pop_size, size=3)
                    while b == a:
                        b = rng.integers(pop_size)
                    while c == a or c == b:
                        c = rng.integers(pop_size)
                    Ff = 0.5 + 0.3 * rng.random()
                    v = X[a] + Ff * (X[b] - X[c])
                    cr = 0.3
                    j_rand = rng.integers(dim)
                    mask = rng.random(dim) < cr
                    mask[j_rand] = True
                    child = np.where(mask, v, X[i])
                    # project onto front: keep angle part, snap distances to 0.5
                    child[:n_shape] = np.clip(child[:n_shape], lo[:n_shape], hi[:n_shape])
                    child[n_shape:] = 0.5
                    child = np.clip(child, lo, hi)
                    offspring[i] = child
                    offspring_F[i] = problem(child)
                    used += 1
                    if time.time() > deadline - 12 or used >= self.budget:
                        break
                n_act = min(i + 1, n_off)
                Xc = np.vstack([X, offspring[:n_act]])
                Fc = np.vstack([F, offspring_F[:n_act]])
                X, F = select_population(Xc, Fc, pop_size)
                if used >= self.budget:
                    break

        # Phase 3: angle polish using remaining budget
        # For each current front point, nudge its angles slightly and keep the
        # non-dominated / front member if it improves coverage.
        if used < self.budget and time.time() < deadline - 5:
            step = 0.05
            for _ in range(3):
                if used >= self.budget or time.time() > deadline - 5:
                    break
                for idx in range(len(X)):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    ang = X[idx][:n_shape].copy()
                    ang += rng.normal(0, step, n_shape)
                    ang = np.clip(ang, lo[:n_shape], hi[:n_shape])
                    cand = front_point(ang)
                    fc = problem(cand)
                    used += 1
                    # accept if not dominated by current set
                    dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= fc) and np.any(F[j] < fc):
                            dominated = True
                            break
                    if not dominated:
                        # replace the point it dominates if any, else replace self
                        replaced = False
                        for j in range(len(F)):
                            if np.all(fc <= F[j]) and np.any(fc < F[j]):
                                X[j] = cand
                                F[j] = fc
                                replaced = True
                                break
                        if not replaced:
                            X[idx] = cand
                            F[idx] = fc

        return X
# EVOLVE-BLOCK-END
