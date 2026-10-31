"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

# Deadline guard: evaluator kills at 150s, we aim to finish well before.
_DEADLINE = 145.0
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _lj_energy(pos):
    """Total Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    r2 = d * d
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    return 4.0 * np.sum(inv_r12 - inv_r6)


def _lj_grad(pos):
    """Gradient of LJ energy with respect to positions."""
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    # Use cdist for pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*(r^-7 - 2*r^-13) / r = 24*(r^-6 - 2*r^-12)/r^2 ... let's do it properly
    # E = 4*(r^-12 - r^-6)
    # dE/dri = 4*(-12*r^-13*dr/dri + 6*r^-7*dr/dri) where dr/dri = -ri/r (for vector from i to j)
    # Actually for pair (i,j): E_ij = 4*(r_ij^-12 - r_ij^-6)
    # dE_ij/dx_i = 4*(-12*r^-13 * dx_i/dr + 6*r^-7 * dx_i/dr)
    # dx_i/dr = -x_ij / r where x_ij = x_i - x_j
    # So dE_ij/dx_i = 4*(-12*r^-13 * (-x_ij/r) + 6*r^-7 * (-x_ij/r))
    #            = 4*(12*r^-14*x_ij - 6*r^-8*x_ij)
    #            = 48*r^-14*x_ij - 24*r^-8*x_ij
    #            = 24*r^-14*x_ij*(2 - r^6) ... let me just compute numerically via vectorized approach
    # 
    # Simpler: dE/dx_i = sum_j dE_ij/dx_i
    # dE_ij/dx_i = 4*(-12*r_ij^-13 * (x_i - x_j)/r_ij + 6*r_ij^-7 * (x_i - x_j)/r_ij)  ... wait sign
    # r_ij = |x_i - x_j|, dr_ij/dx_i = (x_i - x_j)/r_ij
    # dE_ij/dx_i = 4*(-12*r^-13 * (x_i-x_j)/r + 6*r^-7 * (x_i-x_j)/r)
    #            = 4*(x_i-x_j)*(-12*r^-14 + 6*r^-8)
    #            = 24*(x_i-x_j)*(r^-8 - 2*r^-14)
    #            = 24*(x_i-x_j)*r^-14*(r^6 - 2)
    
    coeff = 24.0 * (inv_r6 - 2.0 * inv_r12) / r2  # this is 24*(r^-6 - 2*r^-12)/r^2 = 24*(r^-8 - 2*r^-14)
    # Actually: dE/dx_i = sum_j 24*(r^-8 - 2*r^-14)*(x_i - x_j)
    # But we have diff[i,j] = x_i - x_j
    # So grad[i] = sum_j 24*(r_ij^-8 - 2*r_ij^-14) * diff[i,j]
    # coeff should be 24*(r^-8 - 2*r^-14) = 24*(inv_r6/r2 - 2*inv_r12/r2) = 24*(inv_r6 - 2*inv_r12)/r2
    # Wait: r^-8 = r^-6 / r^2, r^-14 = r^-12 / r^2. Yes.
    
    # Multiply coeff (n,n) with diff (n,n,3), sum over j axis
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    np.fill_diagonal(coeff, 0.0)  # zero out self-interactions
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return grad


def _generate_fcc_truncated_octahedron():
    """
    Generate a truncated octahedron from FCC lattice with 38 atoms.
    The known global minimum for LJ38 is a truncated octahedron with FCC symmetry.
    Structure: layers along [111] direction.
    Layer sizes: 1-5-6-5-1-5-6-5-1 ... that's 39. For 38, we remove one.
    
    Actually the standard LJ38 truncated octahedron has layers: 1-5-6-5-1-5-6-5-1 = 39? No.
    Let me use a different approach: generate FCC points and select the 38 closest to center.
    """
    # FCC lattice vectors
    a = 1.12  # approximate equilibrium spacing
    # Generate a grid of FCC points
    points = []
    # FCC primitive vectors
    v1 = np.array([0, a/np.sqrt(2), a/np.sqrt(2)])
    v2 = np.array([a/np.sqrt(2), 0, a/np.sqrt(2)])
    v3 = np.array([a/np.sqrt(2), a/np.sqrt(2), 0])
    
    # Generate points in a cubic grid
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    
    points = np.array(points)
    # Center
    center = np.mean(points, axis=0)
    points -= center
    
    # Sort by distance from center
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Center again
    points -= np.mean(points, axis=0)
    return points


def _generate_icosahedral():
    """Generate an icosahedral-like structure (local minimum trap, but good starting point for search)."""
    # Two concentric icosahedra: 12+12+12+1 = 37, add 1 more
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    ico /= np.linalg.norm(ico[0])
    
    # Inner shell at r=0.72, outer at r=1.0
    r_inner = 0.72
    r_outer = 1.0
    spacing = 1.12
    
    points = [np.zeros(3)]  # center
    points.extend([r_inner * p * spacing for p in ico])  # 12 inner
    points.extend([r_outer * p * spacing for p in ico])  # 12 outer
    # Need 38 total: 1 + 12 + 12 = 25, need 13 more
    # Add a third shell or duplicate some
    # Let's add 13 more at intermediate positions
    # Actually let's just use 1+12+12+13 = 38
    # Generate 13 more points on a slightly larger shell
    r_third = 1.3
    # Use icosahedron vertices plus center
    extra = [r_third * p * spacing for p in ico]  # 12 more
    extra.append(np.array([0, 0, r_third * spacing]))  # 1 more
    points.extend(extra[:13])
    
    points = np.array(points[:N])
    points -= np.mean(points, axis=0)
    return points


def _relax(pos, maxiter=200, gtol=1e-8):
    """Local relaxation using L-BFGS-B."""
    try:
        res = minimize(
            _lj_energy, pos, jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if res.success or res.fun < _lj_energy(pos):
            return res.x
    except Exception:
        pass
    return pos


def _check_validity(pos):
    """Check minimum separation >= 0.5."""
    d = pdist(pos)
    return d.min() >= 0.5


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy: Start from FCC truncated octahedron, relax, then basin-hop style search
    # with multiple restarts from perturbed FCC structures.
    
    # 1. FCC truncated octahedron (known near-global minimum structure)
    fcc_pos = _generate_fcc_truncated_octahedron()
    if _check_validity(fcc_pos):
        relaxed = _relax(fcc_pos, maxiter=300, gtol=1e-10)
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    else:
        # Fallback: scale to ensure validity
        d = pdist(fcc_pos)
        if d.min() > 0 and d.min() < 0.5:
            scale = 0.5 / d.min() * 1.01
            fcc_pos = fcc_pos * scale
        relaxed = _relax(fcc_pos, maxiter=300, gtol=1e-10)
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # 2. Basin hopping from the best structure found
    if _time_left() < 10:
        # Not enough time, do a final high-precision polish
        if best_pos is not None:
            polished = _relax(best_pos, maxiter=500, gtol=1e-12)
            e = _lj_energy(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished
        return best_pos
    
    # Basin hopping parameters
    T = 0.03
    step_size = 0.3
    n_basins = 0
    rng = np.random.default_rng(42)
    
    while _time_left() > 15:
        # Generate perturbation
        noise = rng.normal(0, step_size, size=best_pos.shape)
        perturbed = best_pos + noise
        
        # Relax the perturbed structure
        relaxed = _relax(perturbed, maxiter=200, gtol=1e-8)
        e_new = _lj_energy(relaxed)
        
        # Metropolis acceptance
        if e_new < best_energy:
            best_energy = e_new
            best_pos = relaxed.copy()
            # Reduce step size when improving
            step_size *= 0.95
            T *= 0.98
        else:
            delta = e_new - best_energy
            accept_prob = np.exp(-delta / T)
            if rng.random() < accept_prob:
                best_pos = relaxed.copy()
        
        n_basins += 1
        
        # Periodically try a larger perturbation to escape local minima
        if n_basins % 20 == 0 and _time_left() > 30:
            big_noise = rng.normal(0, 0.8, size=best_pos.shape)
            big_perturbed = best_pos + big_noise
            big_relaxed = _relax(big_perturbed, maxiter=200, gtol=1e-8)
            e_big = _lj_energy(big_relaxed)
            if e_big < best_energy:
                best_energy = e_big
                best_pos = big_relaxed.copy()
    
    # Final high-precision polish
    if best_pos is not None and _time_left() > 5:
        polished = _relax(best_pos, maxiter=500, gtol=1e-12)
        e = _lj_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pos = polished
    
    # Ensure validity
    if best_pos is not None:
        if not _check_validity(best_pos):
            # Try to fix by scaling
            d = pdist(best_pos)
            if d.min() > 0 and d.min() < 0.5:
                scale = 0.5 / d.min() * 1.01
                best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
