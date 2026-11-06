"""Dzugutov cluster global minima: Parallel basin hopping from icosahedral seeds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 290.0  # Safety margin under 330s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5

def _remaining():
    return TIME_LIMIT - (time.time() - _START)

def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    
    energy = 0.0
    mask1 = r_ij < _a
    if np.any(mask1):
        r1 = r_ij[mask1]
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    mask2 = (r_ij >= _a) & (r_ij < _b)
    if np.any(mask2):
        r2 = r_ij[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return energy

def _grad_fn(pos):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    _, grad = dzugutov_energy_gradient(p)
    return grad.flatten()

def _min_sep_ok(pos, min_sep=_MIN_SEP):
    """Check minimum separation constraint."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    return np.min(r[iu]) >= min_sep

def _enforce_min_sep(pos, min_sep=_MIN_SEP, max_iter=50):
    """Enforce minimum separation by pushing overlapping atoms apart."""
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(max_iter):
        if _min_sep_ok(pos, min_sep):
            break
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_sep:
                    if d < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = (pos[i] - pos[j]) / d
                    push = (min_sep - d) / 2 + 0.005
                    pos[i] += direction * push
                    pos[j] -= direction * push
    return pos

def _get_icosahedral_points():
    """Get icosahedron vertices, edge midpoints, and face centers."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    # 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico[i] - ico[j])
            if 0.9 < d < 1.1:
                edges.append((i, j))
    
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints[:30])
    
    # 20 face centers
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(ico[i] - ico[j])
                djk = np.linalg.norm(ico[j] - ico[k])
                dik = np.linalg.norm(ico[i] - ico[k])
                if (0.9 < dij < 1.1 and 0.9 < djk < 1.1 and 0.9 < dik < 1.1):
                    fc = (ico[i] + ico[j] + ico[k]) / 3
                    fc /= np.linalg.norm(fc)
                    face_centers.append(fc)
    
    unique_fc = []
    for fc in face_centers:
        if not any(np.linalg.norm(fc - ufc) < 0.01 for ufc in unique_fc):
            unique_fc.append(fc)
    face_centers = np.array(unique_fc[:20])
    
    return ico, edge_midpoints, face_centers

def _generate_icosahedral_seed(n, rng, scale=1.0):
    """Generate Mackay icosahedral cluster with n atoms."""
    ico, edge_pts, face_pts = _get_icosahedral_points()
    
    points = []
    remaining = n
    
    # Central atom
    if remaining > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 12 icosahedron vertices
    count1 = min(12, remaining)
    for i in range(count1):
        p = ico[i] * scale * 1.2
        p += rng.normal(0, 0.02, 3)
        points.append(p)
    remaining -= count1
    
    # Second shell: 30 edge midpoints
    if remaining > 0:
        count2 = min(30, remaining)
        for i in range(count2):
            p = edge_pts[i % 30] * scale * 1.9
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count2
    
    # Third shell: 20 face centers
    if remaining > 0:
        count3 = min(20, remaining)
        for i in range(count3):
            p = face_pts[i % 20] * scale * 2.6
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count3
    
    # Fourth shell: more icosahedron vertices at larger radius
    if remaining > 0:
        for i in range(remaining):
            idx = i % 12
            p = ico[idx] * scale * (3.2 + 0.15 * (i // 12))
            p += rng.normal(0, 0.03, 3)
            points.append(p)
    
    points = np.array(points[:n])
    return _enforce_min_sep(points)

def _relax_lbfgsb(pos_flat, maxiter=100):
    """Relax using L-BFGS-B."""
    try:
        result = minimize(
            _energy,
            pos_flat,
            jac=_grad_fn,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20, 'disp': False}
        )
        return result
    except Exception:
        return None

def _basin_hopping_worker(seed, n, rng, result_queue, lock, best_shared):
    """Worker thread for basin hopping."""
    current_pos = seed.copy()
    current_e = _energy(current_pos.flatten())
    best_pos = current_pos.copy()
    best_e = current_e
    
    kick_scale = 0.3
    T = 0.5  # Temperature for Metropolis acceptance
    accept_count = 0
    total_count = 0
    
    while _remaining() > 5.0:
        # Kick
        kick = rng.normal(0, kick_scale, current_pos.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_sep(kicked)
        
        # Relax
        result = _relax_lbfgsb(kicked.flatten(), maxiter=100)
        if result is None:
            continue
        
        new_e = result.fun
        new_pos = result.x.reshape(n, 3)
        
        if not _min_sep_ok(new_pos):
            continue
            
        total_count += 1
        accepted = False
        
        if new_e < current_e:
            accepted = True
        else:
            # Metropolis acceptance
            delta_e = new_e - current_e
            if delta_e < 0:
                accepted = True
            else:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    accepted = True
        
        if accepted:
            accept_count += 1
            current_pos = new_pos
            current_e = new_e
            
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos
                with lock:
                    if best_e < best_shared[0]:
                        best_shared[0] = best_e
                        best_shared[1] = best_pos.copy()
        
        # Adapt kick scale
        if total_count > 10:
            accept_rate = accept_count / total_count
            if accept_rate < 0.3:
                kick_scale *= 1.2
            elif accept_rate > 0.7:
                kick_scale *= 0.8
            kick_scale = np.clip(kick_scale, 0.05, 1.0)
            
            # Reset counters periodically
            if total_count > 50:
                accept_count = 0
                total_count = 0

    # Final relaxation of the best found in this thread
    if best_e < np.inf:
        final_res = _relax_lbfgsb(best_pos.flatten(), maxiter=150)
        if final_res is not None and final_res.fun < best_e:
            best_e = final_res.fun
            best_pos = final_res.x.reshape(n, 3)
            with lock:
                if best_e < best_shared[0]:
                    best_shared[0] = best_e
                    best_shared[1] = best_pos.copy()

def construct(instance):
    """Find low-energy Dzugutov cluster using parallel basin hopping."""
    size = instance["n"]
    
    # Generate diverse seeds
    rng_main = np.random.default_rng(42 + size)
    seeds = []
    
    # Primary icosahedral seed
    seed1 = _generate_icosahedral_seed(size, rng_main, scale=1.0)
    seeds.append(seed1)
    
    # Slightly deformed icosahedral seeds
    for i in range(3):
        scale = rng_main.uniform(0.9, 1.1)
        seed = _generate_icosahedral_seed(size, rng_main, scale=scale)
        seeds.append(seed)
    
    # Random compact seed
    rng_rand = np.random.default_rng(100 + size)
    points = []
    for i in range(size):
        if i == 0:
            points.append(np.array([0.0, 0.0, 0.0]))
        else:
            r = 0.5 + 0.3 * np.sqrt(i)
            theta = rng_rand.uniform(0, 2*np.pi)
            phi = rng_rand.uniform(-np.pi/2, np.pi/2)
            p = np.array([
                r * np.cos(phi) * np.cos(theta),
                r * np.cos(phi) * np.sin(theta),
                r * np.sin(phi)
            ])
            points.append(p)
    random_seed = _enforce_min_sep(np.array(points))
    seeds.append(random_seed)
    
    # Relax all seeds first to get a good starting point
    relaxed_seeds = []
    for seed in seeds:
        if _remaining() < 10.0:
            break
        res = _relax_lbfgsb(seed.flatten(), maxiter=100)
        if res is not None:
            relaxed_seeds.append(res.x.reshape(size, 3))
        else:
            relaxed_seeds.append(seed)
            
    if not relaxed_seeds:
        return seeds[0]
        
    # Setup parallel basin hopping
    n_threads = 4
    best_shared = [np.inf, None]
    lock = threading.Lock()
    threads = []
    
    for i in range(n_threads):
        if i < len(relaxed_seeds):
            seed = relaxed_seeds[i]
        else:
            seed = relaxed_seeds[0] # Reuse best seed if not enough
            
        # Create a separate RNG for each thread
        thread_rng = np.random.default_rng(1000 + i + size)
        
        t = threading.Thread(
            target=_basin_hopping_worker,
            args=(seed, size, thread_rng, None, lock, best_shared)
        )
        threads.append(t)
        t.start()
        
    # Wait for threads to finish or timeout
    for t in threads:
        t.join()
        
    if best_shared[1] is not None:
        return best_shared[1]
    else:
        # Fallback to best relaxed seed
        return relaxed_seeds[0]
# EVOLVE-BLOCK-END
