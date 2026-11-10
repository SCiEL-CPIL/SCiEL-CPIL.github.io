"""Morse cluster global minima: stochastic dynamics + incremental growth + basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

TOTAL_DEADLINE = 290.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster."""
    n = positions.shape[0]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    dvdr = -rho * ex * (ex - 1.0)
    np.fill_diagonal(dvdr, 0.0)
    
    r_safe = np.where(r > 1e-10, r, 1.0)
    r_safe[np.diag_indices(n)] = 1.0
    
    coeff = dvdr / r_safe
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    
    return energy, grad


def _morse_energy_only(positions, rho):
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    if len(positions) <= 1:
        return True
    d = pdist(positions)
    return np.all(d >= min_dist - 1e-6)


def _repair_min_dist(positions, min_dist=0.5, rng=None, max_iter=100):
    if rng is None:
        rng = np.random.default_rng()
    n = positions.shape[0]
    pos = positions.copy().astype(float)
    for _ in range(max_iter):
        if n <= 1:
            break
        d = squareform(pdist(pos))
        np.fill_diagonal(d, 1.0)
        violating = np.argwhere(d < min_dist - 1e-6)
        if len(violating) == 0:
            break
        for i, j in violating:
            if i >= j:
                continue
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                diff = rng.normal(0, 0.01, 3)
                dist = np.linalg.norm(diff)
            direction = diff / dist
            push = (min_dist - d[i, j] + 0.01) / 2.0
            pos[i] += push * direction
            pos[j] -= push * direction
    return pos


def _generate_fcc(n):
    a = 1.0
    basis = [[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]
    L = int(np.ceil((n / 4.0) ** (1.0/3.0))) + 1
    candidates = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i,j,k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= candidates.mean(axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order][:n]


def _generate_bcc(n):
    a = 1.0
    basis = [[0,0,0],[0.5,0.5,0.5]]
    L = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 1
    candidates = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i,j,k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    candidates -= candidates.mean(axis=0)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order][:n]


def _generate_hcp(n):
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    positions = []
    n_layers = int(np.ceil((n / 6.0) ** (1.0/3.0))) + 2
    for layer in range(-n_layers, n_layers+1):
        z = layer * c / 2.0
        if layer % 2 == 0:
            for i in range(-n_layers, n_layers+1):
                for j in range(-n_layers, n_layers+1):
                    x = i*a + (j%2)*a/2.0
                    y = j*a*np.sqrt(3.0)/2.0
                    positions.append([x,y,z])
        else:
            for i in range(-n_layers, n_layers+1):
                for j in range(-n_layers, n_layers+1):
                    x = i*a + a/2.0
                    y = j*a*np.sqrt(3.0)/2.0 + a*np.sqrt(3.0)/6.0
                    positions.append([x,y,z])
    positions = np.array(positions)
    positions -= positions.mean(axis=0)
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    return positions[order][:n]


def _generate_icosahedral(n):
    if n <= 1:
        return np.zeros((n,3))
    phi = (1+np.sqrt(5))/2
    ico_verts = np.array([
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    remaining = n - 1
    r1 = 1.0
    count1 = min(12, remaining)
    for i in range(count1):
        positions.append(r1 * ico_verts[i])
    remaining -= count1
    
    if remaining > 0:
        r2 = 1.8
        count2 = min(60, remaining)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(count2):
            t = (i+0.5)/count2
            z = 1.0 - 2.0*t
            rx = np.sqrt(max(0, 1.0-z*z))
            theta = golden_angle * i
            positions.append(r2 * np.array([rx*np.cos(theta), rx*np.sin(theta), z]))
        remaining -= count2
    
    if remaining > 0:
        r3 = 2.6
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(remaining):
            t = (i+0.5)/remaining
            z = 1.0 - 2.0*t
            rx = np.sqrt(max(0, 1.0-z*z))
            theta = golden_angle * i
            positions.append(r3 * np.array([rx*np.cos(theta), rx*np.sin(theta), z]))
    
    return np.array(positions[:n])


def _generate_shell(n):
    if n <= 1:
        return np.zeros((n,3))
    positions = [np.zeros(3)]
    remaining = n - 1
    shell_idx = 0
    while remaining > 0:
        shell_idx += 1
        if shell_idx == 1:
            count = min(12, remaining)
        elif shell_idx == 2:
            count = min(42, remaining)
        else:
            count = min(int(4*shell_idx*shell_idx), remaining)
        r = shell_idx * 0.55
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(count):
            t = i/count
            z = 1.0 - 2.0*t
            rx = np.sqrt(max(0, 1.0-z*z))
            theta = golden_angle * i
            positions.append(r * np.array([rx*np.cos(theta), rx*np.sin(theta), z]))
        remaining -= count
    return np.array(positions[:n])


def _generate_random_compact(n, rng, min_dist=0.6):
    radius = 1.0 * n**(1.0/3.0) * 0.8
    positions = []
    max_attempts = 10000
    attempts = 0
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in positions:
            if np.linalg.norm(p-q) < min_dist:
                valid = False
                break
        if valid:
            positions.append(p)
    if len(positions) < n:
        positions = []
        step = min_dist * 1.1
        for i in range(-10,11):
            for j in range(-10,11):
                for k in range(-10,11):
                    if len(positions) < n:
                        positions.append(np.array([i,j,k])*step)
    return np.array(positions[:n])


def _relax_lbfgs(positions, rho, max_iter=500, time_limit=15.0):
    start = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    try:
        result = minimize(fun, positions.flatten(), jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': max_iter*20, 'gtol': 1e-8, 'factr': 1e7})
        new_pos = result.x.reshape(-1, 3)
        if _is_valid(new_pos):
            return _morse_energy_only(new_pos, rho), new_pos
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed < time_limit:
        try:
            result = minimize(fun, positions.flatten(), jac=True, method='L-BFGS-B',
                              options={'maxiter': 200, 'maxfun': 4000, 'gtol': 1e-6})
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
        except Exception:
            pass
    
    return _morse_energy_only(positions, rho), positions


def _velocity_verlet_anneal(positions, rho, n_steps=500, dt=0.008,
                            T_start=2.0, T_end=0.02, rng=None, time_limit=15.0):
    if rng is None:
        rng = np.random.default_rng()
    n = positions.shape[0]
    pos = positions.copy().astype(float)
    vel = rng.normal(0, 0.3, pos.shape)
    start_time = time.time()
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        frac = step / max(n_steps-1, 1)
        T = T_start * (T_end/T_start)**frac
        
        e, grad = _morse_energy_and_grad(pos, rho)
        forces = -grad
        vel += 0.5*dt*forces
        pos += dt*vel
        
        target_ke = 1.5*n*T
        current_ke = 0.5*np.sum(vel*vel)
        if current_ke > 1e-10:
            vel *= np.sqrt(target_ke/current_ke)
        
        e2, grad2 = _morse_energy_and_grad(pos, rho)
        forces2 = -grad2
        vel += 0.5*dt*forces2
        
        if not _is_valid(pos, min_dist=0.5):
            pos = _repair_min_dist(pos, min_dist=0.5, rng=rng, max_iter=10)
    
    return pos


def _incremental_growth(n_target, rho, rng, time_limit=20.0):
    """Grow cluster incrementally from small to target size."""
    start_time = time.time()
    current = 5
    positions = _generate_bcc(current)
    
    while current < n_target:
        if time.time() - start_time > time_limit:
            break
        
        # Add one atom at a time
        # Place new atom near existing cluster
        if len(positions) > 0:
            # Try placing near a random existing atom
            idx = rng.integers(0, len(positions))
            base = positions[idx]
            new_atom = base + rng.normal(0, 0.8, 3)
        else:
            new_atom = rng.normal(0, 1.0, 3)
        
        positions = np.vstack([positions, new_atom.reshape(1,3)])
        current += 1
        
        # Relax
        if time.time() - start_time > time_limit:
            break
        energy, positions = _relax_lbfgs(positions, rho, max_iter=100, time_limit=2.0)
        
        if not _is_valid(positions):
            positions = _repair_min_dist(positions, min_dist=0.5, rng=rng)
            if not _is_valid(positions):
                # Remove the last atom and retry
                positions = positions[:-1]
                current -= 1
                continue
    
    return positions[:n_target]


def _basin_hop(positions, rho, rng, n_hops=20, kick_scale=0.3, time_limit=15.0):
    """Basin hopping: perturb, relax, accept if lower energy."""
    start_time = time.time()
    best_energy = _morse_energy_only(positions, rho)
    best_pos = positions.copy()
    
    for hop in range(n_hops):
        if time.time() - start_time > time_limit:
            break
        
        # Kick
        scale = kick_scale * (1.0 + 0.5*np.sin(hop*0.3))
        perturbed = best_pos + rng.normal(0, scale, best_pos.shape)
        if not _is_valid(perturbed):
            perturbed = _repair_min_dist(perturbed, min_dist=0.5, rng=rng)
            if not _is_valid(perturbed):
                continue
        
        # Relax
        time_left = time_limit - (time.time() - start_time)
        if time_left < 1.0:
            break
        energy, relaxed = _relax_lbfgs(perturbed, rho, max_iter=200, time_limit=max(1.0, time_left-1.0))
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    return best_energy, best_pos


def construct(instance):
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    elapsed = time.time() - START_TIME
    remaining = TOTAL_DEADLINE - elapsed
    if remaining < 5.0:
        return _generate_bcc(size)
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    per_instance_budget = min(remaining * 0.95, 48.0)
    instance_start = time.time()
    
    # Phase 1: Stochastic self-assembly from high-T gas
    # Initialize as a gas and let it cool down
    gas_radius = 1.5 * size**(1.0/3.0)
    gas_pos = rng.uniform(-gas_radius, gas_radius, (size, 3))
    # Ensure minimum distance
    gas_pos = _repair_min_dist(gas_pos, min_dist=0.5, rng=rng)
    
    if _is_valid(gas_pos):
        time_left = per_instance_budget * 0.35 - (time.time() - instance_start)
        if time_left > 3.0:
            annealed = _velocity_verlet_anneal(
                gas_pos, rho, n_steps=600, dt=0.006,
                T_start=3.0, T_end=0.01, rng=rng,
                time_limit=min(time_left-1.0, 12.0)
            )
            if _is_valid(annealed):
                energy, relaxed = _relax_lbfgs(annealed, rho, max_iter=300, time_limit=8.0)
                if energy < best_energy and _is_valid(relaxed):
                    best_energy = energy
                    best_positions = relaxed.copy()
    
    # Phase 2: Crystalline templates with relaxation
    generators = [_generate_icosahedral, _generate_fcc, _generate_bcc, _generate_hcp, _generate_shell]
    if size <= 50:
        generators = [_generate_icosahedral, _generate_shell, _generate_fcc, _generate_bcc]
    
    for gen in generators:
        if time.time() - instance_start > per_instance_budget * 0.55:
            break
        try:
            pos = gen(size)
            if len(pos) != size:
                continue
            if not _is_valid(pos):
                pos = _repair_min_dist(pos, min_dist=0.5, rng=rng)
                if not _is_valid(pos):
                    continue
            
            time_left = per_instance_budget * 0.55 - (time.time() - instance_start)
            if time_left < 2.0:
                break
            
            energy, relaxed = _relax_lbfgs(pos, rho, max_iter=400, time_limit=min(time_left-1.0, 12.0))
            if energy < best_energy and _is_valid(relaxed):
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Phase 3: Incremental growth
    if time.time() - instance_start < per_instance_budget * 0.7:
        try:
            time_left = per_instance_budget * 0.7 - (time.time() - instance_start)
            if time_left > 5.0:
                grown = _incremental_growth(size, rho, rng, time_limit=min(time_left-2.0, 15.0))
                if len(grown) == size and _is_valid(grown):
                    energy, relaxed = _relax_lbfgs(grown, rho, max_iter=300, time_limit=8.0)
                    if energy < best_energy and _is_valid(relaxed):
                        best_energy = energy
                        best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Phase 4: Basin hopping from best configuration
    if best_positions is not None and time.time() - instance_start < per_instance_budget * 0.9:
        try:
            time_left = per_instance_budget * 0.9 - (time.time() - instance_start)
            if time_left > 3.0:
                energy, pos = _basin_hop(best_positions, rho, rng, n_hops=30, 
                                         kick_scale=0.25, time_limit=min(time_left-1.0, 15.0))
                if energy < best_energy and _is_valid(pos):
                    best_energy = energy
                    best_positions = pos.copy()
        except Exception:
            pass
    
    # Phase 5: More basin hopping with larger kicks if time remains
    if best_positions is not None and time.time() - instance_start < per_instance_budget:
        try:
            time_left = per_instance_budget - (time.time() - instance_start)
            if time_left > 3.0:
                energy, pos = _basin_hop(best_positions, rho, rng, n_hops=20,
                                         kick_scale=0.4, time_limit=min(time_left-1.0, 12.0))
                if energy < best_energy and _is_valid(pos):
                    best_energy = energy
                    best_positions = pos.copy()
        except Exception:
            pass
    
    # Final relaxation
    if best_positions is not None and time.time() - instance_start < per_instance_budget:
        try:
            time_left = per_instance_budget - (time.time() - instance_start)
            if time_left > 2.0:
                energy, relaxed = _relax_lbfgs(best_positions, rho, max_iter=300, time_limit=min(time_left-0.5, 8.0))
                if energy < best_energy and _is_valid(relaxed):
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    if best_positions is None:
        best_positions = _generate_bcc(size)
        if not _is_valid(best_positions):
            best_positions = _generate_random_compact(size, rng)
        if not _is_valid(best_positions):
            best_positions = _repair_min_dist(best_positions, min_dist=0.5, rng=rng)
        if not _is_valid(best_positions):
            best_positions = np.zeros((size, 3))
            for i in range(size):
                best_positions[i] = rng.uniform(-2, 2, 3)
    
    return best_positions
