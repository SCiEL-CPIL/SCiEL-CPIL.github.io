# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_bcc(n, rng, noise=0.0):
    """Generate a BCC-like cluster structure."""
    # BCC unit cell has 2 atoms. We need roughly n/2 cells.
    # A spherical BCC cluster is a good approximation.
    # Generate points in a BCC lattice and select the closest 'n' to origin.
    
    # Estimate radius needed for n atoms in BCC density (2 atoms per unit volume a^3, a=1)
    # Volume ~ n/2. Radius ~ (3*n/2 / 4*pi)^(1/3)
    r_est = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    
    # Generate lattice points
    # BCC points: (i, j, k) and (i+0.5, j+0.5, k+0.5)
    # Range needs to cover the sphere of radius r_est
    max_idx = int(np.ceil(r_est)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # Corner
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                # Center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
                
    if not points:
        return rng.uniform(-1, 1, size=(n, 3))
        
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # Sort by distance to origin to get a compact cluster
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    if len(sorted_idx) < n:
        # Fallback if we didn't generate enough
        # This shouldn't happen with the margin, but safety first
        extra = n - len(sorted_idx)
        rand_pts = rng.uniform(-r_est, r_est, size=(extra, 3))
        points = np.vstack([points, rand_pts])
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        
    selected = points[sorted_idx[:n]]
    
    if noise > 0:
        selected += rng.normal(0, noise, size=selected.shape)
        
    return selected

def _generate_fcc(n, rng, noise=0.0):
    """Generate an FCC-like cluster structure."""
    # FCC unit cell has 4 atoms.
    r_est = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    
    max_idx = int(np.ceil(r_est)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points (corners + face centers)
                # Corners
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                
                # Face centers
                p2 = np.array([i + 0.5, j + 0.5, k])
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
                
                p3 = np.array([i + 0.5, j, k + 0.5])
                if np.linalg.norm(p3) <= r_est:
                    points.append(p3)
                
                p4 = np.array([i, j + 0.5, k + 0.5])
                if np.linalg.norm(p4) <= r_est:
                    points.append(p4)
                    
    if not points:
        return rng.uniform(-1, 1, size=(n, 3))
        
    points = np.array(points)
    points -= points.mean(axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    if len(sorted_idx) < n:
        extra = n - len(sorted_idx)
        rand_pts = rng.uniform(-r_est, r_est, size=(extra, 3))
        points = np.vstack([points, rand_pts])
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        
    selected = points[sorted_idx[:n]]
    
    if noise > 0:
        selected += rng.normal(0, noise, size=selected.shape)
        
    return selected

def _generate_random_pack(n, rng, min_dist=0.5):
    """Generate a random hard-sphere packing."""
    # Estimate radius
    r_est = 0.7 * n ** (1.0 / 3.0) * 1.5
    pts = []
    attempts = 0
    max_attempts = 50000
    
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-r_est, r_est, 3)
        if np.linalg.norm(p) > r_est:
            continue
            
        # Check distance to existing points
        if pts:
            dists = np.linalg.norm(np.array(pts) - p, axis=1)
            if dists.min() < min_dist:
                continue
                
        pts.append(p)
        
    if len(pts) < n:
        # Fallback to random if packing failed
        pts = [list(rng.uniform(-r_est, r_est, 3)) for _ in range(n)]
        
    return np.array(pts)

def _relax_worker(args):
    """
    Worker function for parallel relaxation.
    Returns (energy, positions).
    """
    x, kind, n, seed_val = args
    rng = np.random.default_rng(seed_val)
    
    try:
        # We use the provided relax function which uses L-BFGS-B
        # We might want to add a small perturbation to break symmetry if needed,
        # but the generators already add noise or use random starts.
        
        e, p = relax(x, 'sutton_chen', kind)
        
        # Validate
        if not np.all(np.isfinite(p)) or not np.isfinite(e):
            return (np.inf, None)
            
        # Check min distance constraint
        if n > 1:
            # Efficient min distance check
            # We can't easily do pairwise without O(n^2) memory if n is large, 
            # but n <= 80 is small enough.
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return (np.inf, None)
                
        return (e, p)
    except Exception:
        return (np.inf, None)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin for 300s limit
    
    # Deterministic seed base
    base_seed = 42 ^ (n * 101)
    
    # Generate diverse starting configurations
    # We will run 4 parallel workers.
    # Let's prepare 4 distinct types of starts.
    
    starts = []
    
    # 1. BCC with low noise
    rng1 = np.random.default_rng(base_seed)
    starts.append(('bcc_low', _generate_bcc(n, rng1, noise=0.05), base_seed))
    
    # 2. FCC with low noise
    rng2 = np.random.default_rng(base_seed + 1)
    starts.append(('fcc_low', _generate_fcc(n, rng2, noise=0.05), base_seed + 1))
    
    # 3. BCC with higher noise (to escape local minima)
    rng3 = np.random.default_rng(base_seed + 2)
    starts.append(('bcc_high', _generate_bcc(n, rng3, noise=0.2), base_seed + 2))
    
    # 4. Random hard-sphere packing
    rng4 = np.random.default_rng(base_seed + 3)
    starts.append(('random', _generate_random_pack(n, rng4), base_seed + 3))
    
    best_e = np.inf
    best_p = None
    
    # Use ThreadPoolExecutor for 4 workers
    # Note: relax might be CPU bound (L-BFGS). Threads in Python might be limited by GIL 
    # if relax doesn't release it. However, if relax uses C extensions (like scipy) that release GIL, 
    # this is effective. If not, it's sequential but still structured.
    # Given the task allows 4 threads, we assume parallelism is beneficial or at least allowed.
    
    # To maximize chance, we can actually submit MORE tasks than threads, 
    # letting the pool handle them. 
    # Let's submit 8 tasks: 4 unique types, and 4 more variations.
    
    tasks = []
    
    # Add 4 more variations
    rng5 = np.random.default_rng(base_seed + 4)
    tasks.append((_generate_fcc(n, rng5, noise=0.2), kind, n, base_seed + 4))
    
    rng6 = np.random.default_rng(base_seed + 5)
    tasks.append((_generate_random_pack(n, rng6, min_dist=0.45), kind, n, base_seed + 5))
    
    rng7 = np.random.default_rng(base_seed + 6)
    # A "shattered" BCC: BCC but with larger noise
    tasks.append((_generate_bcc(n, rng7, noise=0.3), kind, n, base_seed + 6))
    
    rng8 = np.random.default_rng(base_seed + 7)
    # Another random pack
    tasks.append((_generate_random_pack(n, rng8, min_dist=0.55), kind, n, base_seed + 7))
    
    # Add the first 4 to tasks list
    for name, pos, seed in starts:
        tasks.append((pos, kind, n, seed))
        
    # Shuffle tasks to ensure diversity in early results? 
    # Actually, let's just run them.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for t in tasks:
            if time.time() > deadline - 10.0:
                break
            futures.append(executor.submit(_relax_worker, t))
            
        # Process results as they come in
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 5.0:
                break
                
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
                
    # If we still have time and found a good solution, try to refine the best one 
    # with a more aggressive perturbation or just return it.
    # Since relax is already L-BFGS, further refinement might be marginal.
    # However, we can try a few more parallel relaxations starting from the best solution found.
    
    if best_p is not None and time.time() < deadline - 20.0:
        # Try to escape the current best local minimum
        new_tasks = []
        for i in range(4):
            rng_ref = np.random.default_rng(base_seed + 100 + i)
            noise = 0.1 + 0.05 * i
            perturbed = best_p + rng_ref.normal(0, noise, size=best_p.shape)
            new_tasks.append((perturbed, kind, n, base_seed + 100 + i))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_relax_worker, t) for t in new_tasks]
            for future in concurrent.futures.as_completed(futures):
                if time.time() > deadline - 5.0:
                    break
                try:
                    e, p = future.result()
                    if e < best_e:
                        best_e = e
                        best_p = p
                except Exception:
                    pass
                    
    if best_p is None:
        # Fallback: return a simple valid configuration if all failed
        # This should not happen, but safety net
        best_p = np.zeros((n, 3))
        for i in range(n):
            # Grid layout
            x = (i % 10) * 0.5
            y = ((i // 10) % 10) * 0.5
            z = (i // 100) * 0.5
            best_p[i] = [x, y, z]
            
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
