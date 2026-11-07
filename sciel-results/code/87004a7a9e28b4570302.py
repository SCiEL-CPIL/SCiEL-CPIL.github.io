# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _bcc_start(n, rng, spacing=1.0):
    """
    Generate a BCC lattice configuration for n atoms.
    BCC is a strong starting point for SC clusters.
    """
    pts = []
    # Estimate linear size. BCC has 2 atoms per unit cell.
    # Volume ~ n/2. Side ~ (n/2)^(1/3).
    L = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                base = np.array([i, j, k]) * spacing
                pts.append(base)
                pts.append(base + np.array([0.5, 0.5, 0.5]) * spacing)
    
    pts = np.array(pts, dtype=float)
    if len(pts) < n:
        # Fallback to random if lattice is too small (should not happen with +2 margin)
        return rng.uniform(-1, 1, size=(n, 3))
    
    # Center
    pts -= pts.mean(axis=0)
    
    # Select random subset
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]

def _sa_optimize(x0, n, kind, rng, max_steps=300, T0=1.0, Tf=0.01, seed_offset=0):
    """
    Perform Simulated Annealing starting from x0.
    Returns (energy, positions).
    """
    # Initial relaxation to get a good local minimum
    try:
        e_curr, x_curr = relax(np.asarray(x0, dtype=float), 'sutton_chen', kind)
    except Exception:
        return np.inf, x0
    
    if not np.isfinite(e_curr):
        return np.inf, x0
        
    e_best = e_curr
    x_best = x_curr.copy()
    
    # Simulated Annealing parameters
    # Temperature schedule: geometric cooling
    # T(t) = T0 * (Tf/T0)^t
    cool_rate = (Tf / T0) ** (1.0 / max_steps)
    T = T0
    
    # Step size: start moderate, decrease
    # Typical interatomic distance in SC is ~1.0-1.2
    step_scale = 0.5
    
    for i in range(max_steps):
        if T < Tf:
            break
            
        # Perturb
        # Use Gaussian noise
        noise = rng.normal(0, step_scale, size=x_curr.shape)
        x_new = x_curr + noise
        
        # Evaluate new energy
        # We use sutton_chen_energy_gradient to get energy quickly (gradient is ignored)
        # But relax is more robust for the "accept" step? 
        # Actually, SA usually evaluates the raw energy. 
        # Using relax at every step is too slow. 
        # We evaluate raw energy for the Metropolis criterion, and relax only on improvement or at the end?
        # Standard SA: Evaluate E(x_new). If E(x_new) < E(x_curr), accept. Else accept with prob exp(-(E_new-E_curr)/T).
        # However, raw energy evaluation without relaxation might keep us in high-energy regions.
        # Hybrid approach: 
        # 1. Perturb x_curr.
        # 2. Evaluate E(x_new) using sutton_chen_energy_gradient (fast).
        # 3. If accepted, set x_curr = x_new.
        # 4. Periodically (e.g., every 10 steps) or at the end, run relax to clean up.
        
        try:
            e_new, _ = sutton_chen_energy_gradient(x_new, kind)
        except Exception:
            e_new = np.inf
            
        if not np.isfinite(e_new):
            # Reject invalid
            T *= cool_rate
            continue
            
        delta_e = e_new - e_curr
        
        if delta_e < 0:
            # Accept
            x_curr = x_new
            e_curr = e_new
            if e_curr < e_best:
                e_best = e_curr
                x_best = x_curr.copy()
        else:
            # Metropolis criterion
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                x_curr = x_new
                e_curr = e_new
        
        T *= cool_rate
        # Slowly reduce step size
        step_scale *= 0.98
        
        # Periodic relaxation to escape shallow local minima that raw SA might get stuck in?
        # Actually, raw SA can wander far. Relaxing every step is expensive.
        # Let's relax every 20 steps if we have improved recently? 
        # For now, keep it simple: just SA on raw energy.
        # At the end, we will relax the best found.
        
    # Final relaxation of the best configuration found during SA
    try:
        e_final, x_final = relax(x_best, 'sutton_chen', kind)
        if np.isfinite(e_final):
            return e_final, x_final
    except Exception:
        pass
        
    return e_best, x_best

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    # Seed based on instance for reproducibility
    seed = 20240613 ^ (n * 101) ^ (hash(kind) & 0xFFFF)
    rng = np.random.default_rng(seed)
    
    deadline = time.time() + 285.0 # Safety margin
    best_e = np.inf
    best_x = None
    lock = threading.Lock()
    
    def try_candidate(x, tag=""):
        nonlocal best_e, best_x
        if x is None or not np.all(np.isfinite(x)):
            return
        
        # Quick validity check
        if n > 1:
            d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return
                
        # Run SA optimization
        e, p = _sa_optimize(x, n, kind, rng)
        
        if not (np.isfinite(e) and np.all(np.isfinite(p))):
            return
            
        if n > 1:
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return
                
        with lock:
            if e < best_e:
                best_e = e
                best_x = np.array(p, dtype=float)

    # Generate diverse BCC starts with different random subsets and slight distortions
    # We will run multiple independent SA chains in parallel.
    num_chains = 4
    starts = []
    
    for i in range(num_chains):
        # Generate a BCC start
        x_bcc = _bcc_start(n, rng)
        
        # Add some distortion to break symmetry and explore different basins
        # Distortion magnitude increases with chain index
        dist_amp = 0.2 * (i + 1)
        x_distorted = x_bcc + rng.normal(0, dist_amp, size=x_bcc.shape)
        starts.append(x_distorted)
        
        # Also add a pure random start for diversity in one of the chains?
        # Let's make the last chain a random start
        if i == num_chains - 1:
            radius = n ** (1.0/3.0) * 0.8
            x_rand = rng.uniform(-radius, radius, size=(n, 3))
            starts[i] = x_rand

    # Parallel execution
    with ThreadPoolExecutor(max_workers=num_chains) as executor:
        futures = {executor.submit(try_candidate, s): s for s in starts}
        
        for future in as_completed(futures):
            if time.time() > deadline - 10:
                for f in futures:
                    f.cancel()
                break
            try:
                future.result()
            except Exception:
                pass

    # If we have time left, run more SA chains from the best found so far
    # This is a "restart from best" strategy
    if best_x is not None and time.time() < deadline - 20:
        # Run 2 more chains from best_x with larger perturbations
        for i in range(2):
            if time.time() > deadline - 5:
                break
            amp = 0.5 + 0.2 * i
            cand = best_x + rng.normal(0, amp, size=best_x.shape)
            try_candidate(cand)

    # Fallback
    if best_x is None:
        x = np.zeros((n, 3))
        per = int(np.ceil(n ** (1.0 / 3.0))) + 2
        for i in range(n):
            xi = i % per
            yi = (i // per) % per
            zi = i // (per * per)
            x[i] = [xi * 0.5 - per * 0.25, yi * 0.5 - per * 0.25, zi * 0.5]
        try:
            e, p = relax(x, 'sutton_chen', kind)
            if np.isfinite(e):
                best_x = p
        except:
            best_x = rng.uniform(-1, 1, size=(n, 3))

    return np.asarray(best_x, dtype=float)
# EVOLVE-BLOCK-END
