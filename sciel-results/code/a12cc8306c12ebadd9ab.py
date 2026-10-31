import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient: F_i = sum_j 4*(6/r_ij^8 - 12/r_ij^14) * (r_i - r_j)
    # Note: standard LJ force is -dV/dr = 4*(12/r^13 - 6/r^7)
    # dV/dr = 4*(-12/r^13 + 6/r^7)
    # grad_i = sum_j dV/dr * (r_i - r_j)/r = sum_j (4*(-12/r^13 + 6/r^7)/r) * (r_i - r_j)
    # = sum_j 4*(-12/r^14 + 6/r^8) * (r_i - r_j)
    # = sum_j 4*(6/r^8 - 12/r^14) * (r_i - r_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * pos[i] - sum_j coeff[i,j] * pos[j]
    # = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry explicitly.
    Structure: 1 (center) + 12 (1st shell) + 14 (2nd shell) + 11 (3rd shell).
    """
    # Generate FCC points in a box large enough for 38 atoms.
    max_c = 4
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([float(i), float(j), float(k)])
                    
    pts = np.array(pts)
    
    # Calculate distances from origin
    dists = np.linalg.norm(pts, axis=1)
    
    order = np.argsort(dists)
    pts = pts[order][:N]
    
    # Center the cluster
    pts -= np.mean(pts, axis=0)
    
    # Scale to LJ equilibrium distance r_min = 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _make_icosahedral_cluster():
    """
    Generate a 38-atom icosahedral cluster.
    1 center + 12 icosahedron + 25 in outer shell.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    center = np.array([0.0, 0.0, 0.0])
    pts = np.vstack([center, verts])
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    scale = target_r / min_r
    pts *= scale
    
    rng = np.random.default_rng(123)
    r_ico = np.linalg.norm(verts[0]) * scale
    r_shell = 1.6 * r_ico
    
    u = rng.random((25, 2))
    theta = 2 * np.pi * u[:, 0]
    phi_angle = np.arccos(2 * u[:, 1] - 1)
    
    shell_pts = np.column_stack([
        r_shell * np.sin(phi_angle) * np.cos(theta),
        r_shell * np.sin(phi_angle) * np.sin(theta),
        r_shell * np.cos(phi_angle)
    ])
    
    pts = np.vstack([pts, shell_pts])
    pts -= np.mean(pts, axis=0)
    
    return pts

def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    # Use L-BFGS-B with reasonable settings
    maxcor = 50 if gtol < 1e-12 else 20
    
    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxcor": maxcor},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def _de_mutation(best, others, rng, scale=1.0):
    """
    Differential evolution style mutation.
    Generates new candidate by combining best with differences from others.
    """
    n = len(others)
    if n < 2:
        # Fall back to random perturbation
        return best + rng.standard_normal(best.shape) * scale
    
    # Pick two distinct random indices from others
    i1, i2 = rng.choice(n, size=2, replace=False)
    
    # DE/rand/1 mutation: best + F * (other[i1] - other[i2])
    # Scale F to be larger to jump over barriers
    F = scale * (0.8 + 0.4 * rng.random())
    
    candidate = best + F * (others[i1] - others[i2])
    return candidate

def _large_jump_perturb(pos, rng, jump_scale=0.8):
    """
    Large random displacement to escape local minima.
    Specifically targets breaking icosahedral symmetry.
    """
    # Identify atoms that might be in icosahedral positions (high coordination)
    # For simplicity, just apply large random displacement to all atoms
    # with some atoms getting extra displacement
    n_atoms = len(pos)
    
    # Randomly select a subset of atoms to displace more strongly
    n_strong = max(3, n_atoms // 4)
    strong_indices = rng.choice(n_atoms, size=n_strong, replace=False)
    
    noise = rng.standard_normal(pos.shape) * jump_scale
    # Extra displacement for selected atoms
    noise[strong_indices] *= 2.0
    
    return pos + noise

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_pos = None
    best_energy = np.inf
    
    # 1. Initialize with multiple diverse seeds
    seeds = []
    try:
        seeds.append(_make_fcc_cluster())
    except Exception:
        pass
    try:
        seeds.append(_make_icosahedral_cluster())
    except Exception:
        pass
    
    # Add a few random seeds for diversity
    rng_seed = np.random.default_rng(99)
    for _ in range(2):
        try:
            rand_seed = rng_seed.standard_normal((N, 3)) * 2.5
            rand_seed = _ensure_min_sep(rand_seed)
            seeds.append(rand_seed)
        except Exception:
            pass
    
    # Relax all seeds
    relaxed_seeds = []
    for s in seeds:
        try:
            p, e = _relax(s, max_iter=3000, gtol=1e-11, ftol=1e-14)
            relaxed_seeds.append(p)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
            
    if best_pos is None and relaxed_seeds:
        best_pos = relaxed_seeds[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)
    
    # Check time
    if time.time() > deadline - 30.0:
        if best_pos is not None:
            try:
                p, e = _relax(best_pos, max_iter=5000, gtol=1e-14, ftol=1e-18)
                best_pos = p
            except Exception:
                pass
            best_pos = _ensure_min_sep(best_pos)
            return best_pos
        return _make_fcc_cluster()

    # 2. Basin Hopping with DE-style mutations
    rng = np.random.default_rng(42)
    
    # Maintain a population of good solutions for DE mutation
    population = [best_pos.copy()] if best_pos is not None else [relaxed_seeds[0].copy()]
    pop_energies = [best_energy] if best_pos is not None else [best_energy]
    
    max_pop_size = 8
    stagnant_count = 0
    max_stagnant = 8
    
    while time.time() < deadline - 20.0:
        # Choose mutation strategy
        r = rng.random()
        
        if r < 0.4:
            # DE-style mutation from population
            if len(population) >= 2:
                others = [p for p in population if p is not best_pos]
                if len(others) < 1:
                    others = population
                candidate = _de_mutation(best_pos, others, rng, scale=0.3)
            else:
                candidate = best_pos + rng.standard_normal(best_pos.shape) * 0.3
        elif r < 0.7:
            # Moderate random perturbation
            scale = 0.15 + 0.15 * rng.random()
            candidate = best_pos + rng.standard_normal(best_pos.shape) * scale
        elif r < 0.9:
            # Large jump to escape funnel
            candidate = _large_jump_perturb(best_pos, rng, jump_scale=0.6)
        else:
            # Restart from a seed
            if relaxed_seeds:
                candidate = relaxed_seeds[rng.integers(0, len(relaxed_seeds))].copy()
            else:
                candidate = best_pos + rng.standard_normal(best_pos.shape) * 0.5
        
        candidate = _ensure_min_sep(candidate)
        
        # Relax
        try:
            p, e = _relax(candidate, max_iter=2000, gtol=1e-11, ftol=1e-14)
            
            if e < best_energy:
                best_energy = e
                best_pos = p
                stagnant_count = 0
                
                # Update population
                population.append(p.copy())
                pop_energies.append(e)
                
                # Trim population to max size, keeping best solutions
                if len(population) > max_pop_size:
                    # Sort by energy
                    combined = sorted(zip(pop_energies, population), key=lambda x: x[0])
                    population = [x[1] for x in combined[:max_pop_size]]
                    pop_energies = [x[0] for x in combined[:max_pop_size]]
            else:
                stagnant_count += 1
                
                # Occasionally add to population even if not better
                if len(population) < max_pop_size and rng.random() < 0.2:
                    population.append(p.copy())
                    pop_energies.append(e)
        except Exception:
            continue

    # 3. Final High-Precision Polish
    if best_pos is not None:
        remaining = deadline - time.time() - 3.0
        if remaining > 1.0:
            try:
                p, e = _relax(best_pos, max_iter=10000, gtol=1e-15, ftol=1e-18)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
            
            # Second polish if time remains
            remaining = deadline - time.time() - 1.5
            if remaining > 0.5:
                try:
                    p, e = _relax(best_pos, max_iter=5000, gtol=1e-16, ftol=1e-20)
                    if e < best_energy:
                        best_energy = e
                        best_pos = p
                except Exception:
                    pass

    if best_pos is None:
        best_pos = _make_fcc_cluster()
        
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
