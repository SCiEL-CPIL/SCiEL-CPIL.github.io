"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Parallel Tempering (Replica Exchange) Monte Carlo with hard-core exclusion.
Multiple replicas at different temperatures explore the configuration space; swaps
between replicas allow escape from local minima. Each replica maintains a valid
configuration (positions + radii) and is periodically polished with SLSQP.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # safety margin below 150s kill
RNG_SEED = 42


def _elapsed(t0):
    return time.time() - t0


def _remaining(t0):
    return DEADLINE - time.time()


def _validate(pos, rad, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    n = len(rad)
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], rad[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - tol:
                return False
    return True


def _max_radius_for(pos, rad):
    """Maximal radius for each circle given current positions and other radii."""
    n = len(rad)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - rad[j])
        mr[i] = max(0.0, dmin)
    return mr


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return edges
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _slsqp_polish(pos, rad, maxiter=80, t0=None):
    """Quick SLSQP polish using Delaunay edges only."""
    if t0 is not None and _remaining(t0) < 5:
        return pos, rad

    pos0 = pos.copy()
    rad0 = np.maximum(rad, 0.0)
    x0 = np.concatenate([pos0.reshape(-1), rad0])

    arr = np.column_stack([pos0, rad0])
    if not _validate(pos0, rad0):
        rad0 = np.maximum(rad0 * 0.98, 0.0)
        x0 = np.concatenate([pos0.reshape(-1), rad0])
        if not _validate(pos0, rad0):
            return pos0, rad0

    edges = _delaunay_edges(pos0)
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})

    def obj(x):
        return -np.sum(x[2*N:])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-9})
        if res.success or np.sum(res.x[2*N:]) > np.sum(rad0):
            pos_new = res.x[:2*N].reshape(N, 2)
            rad_new = np.maximum(res.x[2*N:], 0.0)
            if _validate(pos_new, rad_new):
                return pos_new, rad_new
    except Exception:
        pass
    return pos0, rad0


def _initial_grid(side=5):
    """Grid initialization."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    rad = np.full(N, 0.5 / side - 1e-4)
    rad = _max_radius_for(pos, rad)
    return pos, rad


def _initial_hex():
    """Hexagonal packing initialization."""
    hex_pos = []
    r_hex = 0.11
    dx = 2 * r_hex
    dy = r_hex * np.sqrt(3)
    for row in range(9):
        for col in range(9):
            x = col * dx + (row % 2) * dx / 2 + 0.12
            y = row * dy + 0.12
            if 0.03 < x < 0.97 and 0.03 < y < 0.97:
                hex_pos.append((x, y))
    hex_pos = hex_pos[:N]
    if len(hex_pos) < N:
        return _initial_grid(5)
    hp = np.array(hex_pos, dtype=float)
    hr = _max_radius_for(hp, np.full(N, 0.08))
    return hp, hr


def _initial_random(rng):
    """Random initialization with small circles."""
    p0 = rng.random((N, 2)) * 0.7 + 0.15
    r0 = np.full(N, 0.03)
    r0 = _max_radius_for(p0, r0)
    return p0, r0


def _mc_move(pos, rad, T, rng, move_size=0.03):
    """
    One Monte Carlo move: propose a random perturbation to one circle's position
    and/or radius. Accept if the new configuration is valid (hard-core).
    At high T, also allow small violations (penalized acceptance).
    Returns (pos, rad, accepted).
    """
    i = rng.integers(N)
    move_type = rng.random()
    
    pos_new = pos.copy()
    rad_new = rad.copy()
    
    if move_type < 0.7:
        # Position move
        dx = rng.normal(0, move_size)
        dy = rng.normal(0, move_size)
        pos_new[i, 0] += dx
        pos_new[i, 1] += dy
    elif move_type < 0.9:
        # Radius grow
        delta_r = rng.uniform(0, move_size * 0.5)
        rad_new[i] += delta_r
    else:
        # Radius shrink
        delta_r = rng.uniform(0, move_size * 0.3)
        rad_new[i] = max(0.0, rad_new[i] - delta_r)

    # Check validity (hard-core)
    if _validate(pos_new, rad_new):
        return pos_new, rad_new, True
    
    # At high temperature, allow small violations with Boltzmann probability
    # Compute violation magnitude
    violation = 0.0
    for k in range(N):
        x, y, r = pos_new[k, 0], pos_new[k, 1], rad_new[k]
        violation = max(violation, max(0, r - x), max(0, x - (1 - r)),
                        max(0, r - y), max(0, y - (1 - r)))
    for k in range(N):
        for j in range(k + 1, N):
            d = np.hypot(pos_new[k, 0] - pos_new[j, 0], pos_new[k, 1] - pos_new[j, 1])
            violation = max(violation, rad_new[k] + rad_new[j] - d)
    
    if T > 0 and violation < 0.01:
        # Accept with probability exp(-violation / T)
        if rng.random() < np.exp(-violation / T):
            # Repair: shrink radius slightly
            rad_new = np.maximum(rad_new * 0.995, 0)
            if _validate(pos_new, rad_new):
                return pos_new, rad_new, True
    
    return pos, rad, False


def _parallel_tempering(t0, n_replicas=8, n_steps=3000, swap_interval=50):
    """
    Parallel Tempering Monte Carlo.
    Returns the best (pos, rad) found across all replicas.
    """
    rng = np.random.default_rng(RNG_SEED)
    
    # Initialize replicas with diverse starting points
    init_funcs = [
        lambda: _initial_grid(5),
        lambda: _initial_grid(6),
        lambda: _initial_hex(),
        lambda: _initial_random(rng),
        lambda: _initial_random(rng),
        lambda: _initial_grid(7),
        lambda: _initial_random(rng),
        lambda: _initial_hex(),
    ]
    
    # Temperature schedule: from high (exploratory) to low (exploitative)
    T_max = 0.05
    T_min = 0.001
    temperatures = np.geomspace(T_max, T_min, n_replicas)
    
    replicas = []
    for k in range(n_replicas):
        pos_k, rad_k = init_funcs[k % len(init_funcs)]()
        replicas.append({'pos': pos_k, 'rad': rad_k, 'T': temperatures[k]})
    
    best_score = -1.0
    best_pos = None
    best_rad = None
    
    def update_best(pos, rad):
        nonlocal best_score, best_pos, best_rad
        s = float(np.sum(rad))
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad.copy()
    
    # Initial best
    for rep in replicas:
        update_best(rep['pos'], rep['rad'])
    
    step = 0
    while _remaining(t0) > 15:
        step += 1
        
        # --- MC moves for each replica ---
        for k in range(n_replicas):
            rep = replicas[k]
            T = rep['T']
            # Adaptive move size: larger at higher T
            move_size = 0.01 + 0.03 * (T / T_max)
            pos_new, rad_new, accepted = _mc_move(
                rep['pos'], rep['rad'], T, rng, move_size
            )
            if accepted:
                rep['pos'] = pos_new
                rep['rad'] = rad_new
                # Update radii to max feasible
                rep['rad'] = _max_radius_for(rep['pos'], rep['rad'])
                update_best(rep['pos'], rep['rad'])
        
        # --- Replica swap (Metropolis criterion) ---
        if step % swap_interval == 0:
            for k in range(n_replicas - 1):
                if _remaining(t0) < 15:
                    break
                rep_a = replicas[k]
                rep_b = replicas[k + 1]
                T_a = rep_a['T']
                T_b = rep_b['T']
                
                # Energy = -sum(radii) (we want to maximize sum of radii)
                E_a = -np.sum(rep_a['rad'])
                E_b = -np.sum(rep_b['rad'])
                
                # Swap energy change
                dE = (E_a / T_b + E_b / T_a) - (E_a / T_a + E_b / T_b)
                
                if dE <= 0 or rng.random() < np.exp(-dE):
                    # Perform swap
                    replicas[k]['pos'], replicas[k+1]['pos'] = replicas[k+1]['pos'], replicas[k]['pos']
                    replicas[k]['rad'], replicas[k+1]['rad'] = replicas[k+1]['rad'], replicas[k]['rad']
                    replicas[k]['T'], replicas[k+1]['T'] = replicas[k+1]['T'], replicas[k]['T']
        
        # --- Periodic SLSQP polish on best replica ---
        if step % 200 == 0 and _remaining(t0) > 20:
            # Find best replica
            best_k = max(range(n_replicas), key=lambda k: np.sum(replicas[k]['rad']))
            rep = replicas[best_k]
            pos_p, rad_p = _slsqp_polish(rep['pos'], rep['rad'], maxiter=50, t0=t0)
            if np.sum(rad_p) > np.sum(rep['rad']):
                rep['pos'] = pos_p
                rep['rad'] = rad_p
                update_best(rep['pos'], rep['rad'])
    
    # Final polish on the best solution
    if best_pos is not None and _remaining(t0) > 5:
        best_pos, best_rad = _slsqp_polish(best_pos, best_rad, maxiter=100, t0=t0)
        update_best(best_pos, best_rad)
    
    if best_pos is None:
        pos, rad = _initial_grid(5)
        return pos, rad
    return best_pos, best_rad


def construct():
    t0 = time.time()
    
    # Run parallel tempering
    pos, rad = _parallel_tempering(t0, n_replicas=8, n_steps=5000, swap_interval=40)
    
    # Final validation and repair
    arr = np.column_stack([pos, rad])
    if not _validate(pos, rad):
        rad = np.maximum(rad * 0.999, 0)
        if not _validate(pos, rad):
            rad = np.maximum(rad * 0.99, 0)
            if not _validate(pos, rad):
                # Fallback to grid
                side = 5
                cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                         for i in range(side) for j in range(side)]
                return np.array(cells[:N])
    
    # One more polish if time allows
    if _remaining(t0) > 10:
        pos, rad = _slsqp_polish(pos, rad, maxiter=100, t0=t0)
        arr = np.column_stack([pos, rad])
        if not _validate(pos, rad):
            rad = np.maximum(rad * 0.999, 0)
    
    return np.column_stack([pos, rad])


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
