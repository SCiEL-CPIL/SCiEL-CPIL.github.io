# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_crystalline_starts(n, kind, rng):
    """
    Generate diverse starting configurations based on crystalline structures.
    Returns a list of (n, 3) arrays.
    """
    starts = []
    
    # Define lattice generators
    def fcc_lattice(count, spacing=1.0):
        # FCC points in unit cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
        pts = []
        # Generate a grid of unit cells
        # Estimate linear size
        L = int(np.ceil((count / 4.0) ** (1.0/3.0))) + 2
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    base = np.array([i, j, k]) * spacing
                    pts.append(base + np.array([0, 0, 0]) * spacing)
                    pts.append(base + np.array([0, 0.5, 0.5]) * spacing)
                    pts.append(base + np.array([0.5, 0, 0.5]) * spacing)
                    pts.append(base + np.array([0.5, 0.5, 0]) * spacing)
        pts = np.array(pts)
        if len(pts) < count:
            return None
        # Center
        pts -= pts.mean(axis=0)
        # Random subset
        idx = rng.choice(len(pts), count, replace=False)
        return pts[idx]

    def bcc_lattice(count, spacing=1.0):
        pts = []
        L = int(np.ceil((count / 2.0) ** (1.0/3.0))) + 2
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    base = np.array([i, j, k]) * spacing
                    pts.append(base)
                    pts.append(base + np.array([0.5, 0.5, 0.5]) * spacing)
        pts = np.array(pts)
        if len(pts) < count:
            return None
        pts -= pts.mean(axis=0)
        idx = rng.choice(len(pts), count, replace=False)
        return pts[idx]

    def hcp_lattice(count, spacing=1.0):
        # HCP is more complex. Approximate with a hexagonal grid of layers.
        pts = []
        a = spacing
        c = 1.633 * a # ideal c/a ratio
        L = int(np.ceil((count / 3.0) ** (1.0/3.0))) + 2
        # Generate hexagonal points in 2D
        for k in range(-L, L+1):
            z = k * c
            for i in range(-L, L+1):
                for j in range(-L, L+1):
                    if i + j + k > 0 or i + j + k < 0:
                        # Hexagonal coordinates q, r
                        x = a * (i + 0.5 * j)
                        y = a * (np.sqrt(3) / 2 * j)
                        # Two layers for HCP: layer A and layer B
                        if k % 2 == 0:
                            # Layer A
                            pts.append([x, y, z])
                        else:
                            # Layer B (shifted)
                            pts.append([x + a/2, y + a*np.sqrt(3)/6, z])
                            # Actually HCP stacking is ABAB.
                            # Let's just use a simple hexagonal close packing approximation
                            # Or just use FCC which is often a good proxy.
                            # To keep it simple and robust, I'll stick to FCC and BCC primarily,
                            # and add a "random dense" start.
                            pass
        # This HCP generation is messy. Let's skip it and use FCC, BCC, and Simple Cubic.
        # Simple Cubic
        pts_sc = []
        L_sc = int(np.ceil(count ** (1.0/3.0))) + 2
        for i in range(-L_sc, L_sc+1):
            for j in range(-L_sc, L_sc+1):
                for k in range(-L_sc, L_sc+1):
                    pts_sc.append([i, j, k] * spacing)
        pts_sc = np.array(pts_sc)
        if len(pts_sc) >= count:
             pts_sc -= pts_sc.mean(axis=0)
             idx = rng.choice(len(pts_sc), count, replace=False)
             starts.append(pts_sc[idx])
        
        # We'll return starts from FCC and BCC in the main loop to keep this function clean.
        return None # Placeholder, we handle FCC/BCC outside or in a unified way.

    # Better approach: Generate specific types
    types = ['fcc', 'bcc', 'sc', 'random_dense']
    
    for t in types:
        if t == 'fcc':
            pts = fcc_lattice(n, spacing=1.0)
            if pts is not None:
                starts.append(pts)
        elif t == 'bcc':
            pts = bcc_lattice(n, spacing=1.0)
            if pts is not None:
                starts.append(pts)
        elif t == 'sc':
            pts = []
            L = int(np.ceil(n ** (1.0/3.0))) + 2
            for i in range(-L, L+1):
                for j in range(-L, L+1):
                    for k in range(-L, L+1):
                        pts.append([i, j, k])
            pts = np.array(pts, dtype=float)
            if len(pts) >= n:
                pts -= pts.mean(axis=0)
                idx = rng.choice(len(pts), n, replace=False)
                starts.append(pts[idx])
        elif t == 'random_dense':
            # Random points in a sphere, ensuring no overlaps
            radius = n ** (1.0/3.0) * 0.8
            pts = []
            attempts = 0
            while len(pts) < n and attempts < 10000:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if pts:
                    dists = np.linalg.norm(np.array(pts) - p, axis=1)
                    if np.min(dists) < 0.5:
                        continue
                pts.append(p)
                attempts += 1
            if len(pts) == n:
                starts.append(np.array(pts))
                
    return starts

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    parameter = kind if kind is not None else instance.get("rho")
    
    # Seed based on instance for reproducibility
    seed = 20240613 ^ (n * 101) ^ (hash(kind) & 0xFFFF)
    rng = np.random.default_rng(seed)
    
    deadline = time.time() + 285.0 # Safety margin
    best_e = np.inf
    best_x = None
    lock = __import__('threading').Lock()

    def try_relax(x):
        nonlocal best_e, best_x
        if x is None or not np.all(np.isfinite(x)):
            return
        # Check validity
        if n > 1:
            d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return
        
        try:
            e, p = relax(np.asarray(x, dtype=float), 'sutton_chen', parameter)
        except Exception:
            return
        
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

    # 1. Generate structured starts
    starts = _generate_crystalline_starts(n, kind, rng)
    
    # 2. Add a few random perturbations of the best structured start if we have time
    # But first, let's just run the structured ones in parallel.
    
    # We can also generate "perturbed" versions of the FCC/BCC starts
    for i in range(min(4, len(starts))):
        if time.time() > deadline - 10:
            break
        base = starts[i]
        # Add small noise
        noisy = base + rng.normal(0, 0.1, size=base.shape)
        starts.append(noisy)

    # Parallel relaxation
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(try_relax, s): s for s in starts}
        
        # As they complete, check time. If we have plenty of time left, generate more?
        # For now, let them finish.
        for future in as_completed(futures):
            if time.time() > deadline - 5:
                # Cancel pending
                for f in futures:
                    f.cancel()
                break
            try:
                future.result()
            except Exception:
                pass

    # 3. If we still have time, do local search / perturbation around the best found
    if best_x is not None and time.time() < deadline - 15:
        guard = 0
        while time.time() < deadline - 5 and guard < 50:
            guard += 1
            # Basin hopping style: perturb best, relax
            amp = 0.1 + 0.05 * (guard % 5)
            cand = best_x + rng.normal(0, amp, size=best_x.shape)
            try_relax(cand)
            
    # Fallback if nothing worked
    if best_x is None:
        # Simple grid
        x = np.zeros((n, 3))
        per = int(np.ceil(n ** (1.0 / 3.0))) + 2
        for i in range(n):
            xi = i % per
            yi = (i // per) % per
            zi = i // (per * per)
            x[i] = [xi * 0.5 - per * 0.25, yi * 0.5 - per * 0.25, zi * 0.5]
        try_relax(x)
        if best_x is None:
            # Last resort random
            best_x = rng.uniform(-1, 1, size=(n, 3))

    return np.asarray(best_x, dtype=float)
# EVOLVE-BLOCK-END
