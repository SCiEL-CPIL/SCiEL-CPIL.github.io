"""Morse cluster global minima: Parallel Basin Hopping with diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 285.0  # Safety margin before 300s kill
_DEADLINE = _START + _BUDGET


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0


def _validate_and_fix(coords):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        # Simple iterative repulsion
        for _ in range(100):
            d = cdist(coords, coords)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            if min_d >= 0.5:
                break
            # Find closest pair
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            if i == j:
                continue
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.array([0.01, 0.01, 0.01])
                dist = np.linalg.norm(diff)
            diff /= dist
            coords[i] += 0.01 * diff
            coords[j] -= 0.01 * diff
    return coords


def _perturb(coords, rng, scale):
    """Random displacement of a random subset of atoms."""
    n = len(coords)
    k = max(1, int(n * 0.2)) # Perturb 20% of atoms
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.normal(0.0, scale, size=(k, 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _basin_hopping_chain(rho, x0, rng, max_iter=100, temp_start=0.1, temp_end=0.01):
    """
    Basin hopping with cooling temperature.
    Returns (best_x, best_e).
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    e_rel, x_rel = relax(x0, 'morse', rho)
    if e_rel < 1e8:
        cur_x = x_rel
        cur_e = e_rel
        if cur_e < best_e:
            best_e = cur_e
            best_x = x_rel.copy()

    for i in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Cooling schedule
        t = i / max_iter
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Perturb
        # Scale increases slightly as we cool to allow larger jumps initially? 
        # Actually, usually scale is constant or increases. Let's use a moderate scale.
        scale = 0.2 + 0.1 * np.sin(np.pi * t) # Vary scale a bit
        x_new = _perturb(cur_x, rng, scale)
        
        # Relax
        e_rel, x_rel = relax(x_new, 'morse', rho)
        
        if e_rel > 1e8:
            # Try to fix and retry once
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel > 1e8:
                continue

        dE = e_rel - cur_e
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
                
    return best_x, best_e


def _generate_seeds(n, rng):
    """Generate a diverse set of initial configurations."""
    seeds = []
    
    # 1. FCC-like packing
    # Generate a large FCC lattice and take the n closest to origin
    a = 1.0
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    fcc_seed = pts[order[:n]].copy()
    fcc_seed += rng.normal(0, 0.05, fcc_seed.shape)
    seeds.append(fcc_seed)

    # 2. Icosahedral shell
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    if n <= 12:
        ico_seed = ico[:n].copy()
    else:
        # Add inner atoms
        inner_n = n - 12
        inner_r = 0.6
        inner_pts = []
        for _ in range(inner_n):
            # Random points in sphere
            while True:
                p = rng.uniform(-inner_r, inner_r, 3)
                if np.linalg.norm(p) < inner_r:
                    break
            inner_pts.append(p)
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.05, ico_seed.shape)
    seeds.append(ico_seed)

    # 3. Random Compact
    radius = n ** (1/3) * 0.8
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 4. Spherical Shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.5
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.05, shell_seed.shape)
    seeds.append(shell_seed)

    # 5. Two-Shell Structure
    if n > 10:
        outer_n = max(10, int(n * 0.6))
        inner_n = n - outer_n
        outer_r = 1.2
        inner_r = 0.5
        
        outer_pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            outer_pts.append([rr * np.cos(theta), rr * np.sin(theta), z])
        outer_pts = np.array(outer_pts) * outer_r
        
        inner_pts = rng.uniform(-inner_r, inner_r, size=(inner_n, 3))
        two_shell = np.vstack([outer_pts, inner_pts])
        two_shell += rng.normal(0, 0.05, two_shell.shape)
        seeds.append(two_shell)

    return seeds


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    # Relax all seeds first to get a baseline
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        e, x = relax(seed, 'morse', rho)
        if e < 1e8:
            if e < best_e:
                best_e = e
                best_x = x.copy()
        else:
            # Try to fix
            x_fixed = _validate_and_fix(x)
            e, x = relax(x_fixed, 'morse', rho)
            if e < 1e8:
                if e < best_e:
                    best_e = e
                    best_x = x.copy()

    if best_x is None:
        # Fallback
        return seeds[0]

    # Parallel Basin Hopping
    # We will run multiple chains in parallel (sequentially in code, but logically distinct)
    # To maximize efficiency, we will run chains for a fixed time slice, then check best.
    
    num_chains = len(seeds) + 2 # Add a couple of extra random starts
    if num_chains < 5:
        num_chains = 5
    
    chain_starts = seeds[:]
    for _ in range(num_chains - len(seeds)):
        r = n ** (1/3) * 0.8
        chain_starts.append(rng.uniform(-r, r, size=(n, 3)))
    
    # We will iterate through chains, giving each a portion of the time
    # Time per chain = _remaining() / num_chains
    time_per_chain = _remaining() / num_chains
    
    for i in range(num_chains):
        if _remaining() < 5.0:
            break
            
        # If time per chain is too small, break
        if time_per_chain < 1.0:
            break
            
        x0 = chain_starts[i].copy()
        
        # Run basin hopping with limited time
        # We estimate max_iter based on time
        # Assume each relax+perturb takes ~0.01s for small n, ~0.1s for large n
        # Let's just run until time is up or max_iter
        max_iter = 50
        x_bh, e_bh = _basin_hopping_chain(rho, x0, rng, max_iter=max_iter, temp_start=0.1, temp_end=0.01)
        
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh.copy()

    # Final Polish
    if _remaining() > 5.0:
        # Do a few more relaxations on the best found
        for _ in range(5):
            if _remaining() < 2.0:
                break
            # Small perturbation and relax
            x_pert = _perturb(best_x, rng, 0.05)
            e, x = relax(x_pert, 'morse', rho)
            if e < 1e8 and e < best_e:
                best_e = e
                best_x = x.copy()
        
        # One final relax
        _, best_x = relax(best_x, 'morse', rho)
        
        # Validate
        if _morse_energy(rho, best_x) < 1e8:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
