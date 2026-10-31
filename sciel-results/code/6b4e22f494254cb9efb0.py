import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    shifted = np.clip(shifted, 0, 100)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(x_flat, temperature):
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(x_flat, temperature):
    s = _soft_score(x_flat, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Epigraph formulation: maximize t subject to all triangle areas >= t * hull_area ----------
# We reformulate: minimize -t where t is the minimum normalized triangle area.
# Use a smooth approximation: minimize sum of soft-penalties for triangles below target.

def _neg_score_epigraph(x_flat, t_target):
    """
    Penalty-based objective: penalize triangles with area < t_target * hull_area.
    This is a smooth surrogate that helps escape local minima.
    """
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    normalized = areas / hull_area
    # Penalty: sum of (t_target - normalized_i)^+ squared
    deficit = np.maximum(t_target - normalized, 0.0)
    penalty = np.sum(deficit ** 2)
    # Also reward increasing the minimum
    min_norm = float(np.min(normalized))
    return penalty - 0.01 * min_norm


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1)]),
        np.column_stack([r2 * np.cos(a2)]),
        np.column_stack([r3 * np.cos(a3)]),
    ])
    # Fix: need both x and y
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    rng = np.random.RandomState(42)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_random_ring(rng):
    n_outer = int(rng.integers(5, 10))
    n_inner = N - n_outer
    if n_inner > 0:
        o_angles = np.sort(rng.uniform(0, 2 * np.pi, n_outer))
        i_angles = np.sort(rng.uniform(0, 2 * np.pi, n_inner))
        r_in = rng.uniform(0.3, 0.7)
        o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
        i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
        return np.vstack([o, i])
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=12):
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 8:
            break

        if time.time() < deadline - 6:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

    return best, best_s


def _polish_true_objective(x0, deadline):
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.02, sigma_end=0.001):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)

    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break

        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress

        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction

            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break

    return best, best_s


# ---------- New: Differential Evolution + polish ----------

def _differential_evolution_objective(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _de_optimize(x0, deadline, pop_size=40, max_iter=200, mutation=(0.5, 1.0), crossover=0.7):
    """
    Lightweight differential evolution for the 26-dimensional point space.
    """
    n = len(x0)
    rng = np.random.default_rng(7)
    
    # Initialize population
    pop = np.array([x0.copy()])
    for i in range(pop_size - 1):
        pop_i = x0 + rng.normal(0, 0.1, n)
        pop = np.vstack([pop, pop_i])
    
    fitnesses = np.array([_differential_evolution_objective(p) for p in pop])
    best_idx = np.argmin(fitnesses)
    best_x = pop[best_idx].copy()
    best_f = fitnesses[best_idx]
    
    for it in range(max_iter):
        if time.time() >= deadline - 10:
            break
        
        new_pop = [pop[best_idx].copy()]
        new_fitnesses = [best_f]
        
        for i in range(pop_size - 1):
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while c == best_idx:
                c = rng.randint(pop_size)
            
            # Mutation
            F = rng.uniform(*mutation)
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            CR = crossover
            r = rng.random()
            j_rand = rng.randint(n)
            trial = np.where(rng.random(n) < CR, v, pop[i])
            trial[j_rand] = v[j_rand]
            
            # Evaluate
            f_trial = _differential_evolution_objective(trial)
            
            if f_trial <= fitnesses[i]:
                new_pop.append(trial)
                new_fitnesses.append(f_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = len(new_pop) - 1
            else:
                new_pop.append(pop[i].copy())
                new_fitnesses.append(fitnesses[i])
        
        pop = np.array(new_pop)
        fitnesses = np.array(new_fitnesses)
    
    return best_x, -best_f


# ---------- New: Coordinate descent with adaptive steps ----------

def _coordinate_descent(points, deadline, n_rounds=50):
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(99)
    
    for rnd in range(n_rounds):
        if time.time() >= deadline - 5:
            break
        
        improved = False
        for idx in range(N):
            if time.time() >= deadline - 5:
                break
            
            # Try small perturbations in x and y
            for dim in range(2):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    cand = best.copy()
                    cand[idx, dim] += delta
                    s = _score(cand)
                    if s > best_s + 1e-15:
                        best_s = s
                        best = cand.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best, best_s


# ---------- New: Simulated annealing ----------

def _simulated_annealing(points, deadline, T_start=0.05, T_end=0.001, n_iter=500):
    best = points.copy()
    best_s = _score(best)
    current = points.copy()
    current_s = _score(current)
    rng = np.random.RandomState(77)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Random perturbation
        idx = rng.randint(N)
        delta = rng.randn(2) * 0.02
        cand = current.copy()
        cand[idx] += delta
        
        cand_s = _score(cand)
        
        if cand_s > current_s or rng.random() < np.exp((cand_s - current_s) / T):
            current = cand
            current_s = cand_s
            if cand_s > best_s:
                best_s = cand_s
                best = cand.copy()
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s kill

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = [
        _seed_regular(),
        _seed_two_ring(7, 0.55, np.pi / 6),
        _seed_two_ring(6, 0.5, np.pi / 5),
        _seed_two_ring(8, 0.4, np.pi / 4),
        _seed_three_ring(5, 4, 4, 0.65, 0.35, np.pi / 5, np.pi / 4),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, np.pi / 3, np.pi / 6),
        _seed_quasi_regular(),
    ]

    # Add randomized ring-based seeds
    for _ in range(8):
        seeds.append(_seed_random_ring(rng))

    # Evaluate all seeds
    for seed in seeds:
        _update_best(seed)

    # Annealed optimization from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                seed.flatten(),
                deadline,
                max_temp=0.05,
                min_temp=1e-4,
                n_steps=10,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # Differential Evolution from best and a few seeds
    if best_points is not None and time.time() < deadline - 40:
        try:
            de_x, de_s = _de_optimize(best_points.flatten(), deadline, pop_size=30, max_iter=100)
            _update_best(de_x.reshape(N, 2))
        except Exception:
            pass

    # DE from a couple of different seeds
    for seed_idx in [0, 2, 4]:
        if time.time() > deadline - 35:
            break
        try:
            de_x, de_s = _de_optimize(seeds[seed_idx].flatten(), deadline, pop_size=25, max_iter=80)
            _update_best(de_x.reshape(N, 2))
        except Exception:
            pass

    # Polish the best found so far
    if best_points is not None and time.time() < deadline - 25:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    # Coordinate descent
    if best_points is not None and time.time() < deadline - 15:
        try:
            cd_points, cd_s = _coordinate_descent(best_points, deadline, n_rounds=30)
            _update_best(cd_points)
        except Exception:
            pass

    # Local search from best
    if best_points is not None and time.time() < deadline - 12:
        try:
            ls_points, s_ls = _local_search(best_points, deadline, n_iter=200)
            _update_best(ls_points)
        except Exception:
            pass

    # Simulated annealing
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_points, sa_s = _simulated_annealing(best_points, deadline, n_iter=300)
            _update_best(sa_points)
        except Exception:
            pass

    # Final hard-objective polish
    if best_points is not None and time.time() < deadline - 8:
        try:
            x_final, s_final = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_final.reshape(N, 2))
        except Exception:
            pass

    # More annealed restarts if time permits
    while time.time() < deadline - 8:
        seed_idx = int(rng.integers(0, len(seeds)))
        base_seed = seeds[seed_idx].copy()
        noise = rng.normal(0, 0.05, (N, 2))
        perturbed = base_seed + noise
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                perturbed.flatten(),
                deadline,
                max_temp=0.03,
                min_temp=1e-4,
                n_steps=8,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # One more polish if we improved
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
