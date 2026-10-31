"""LJ38 cluster global minimum search using symmetry-adapted FCC truncated octahedron + basin hopping."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = time.time() + 110.0  # safety margin before 120s limit


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions array of shape (N, 3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** -3
    r12 = r6 * r6
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    e = 4.0 * (r12 - r6)
    e[0, 0] = 0.0
    energy = 0.5 * np.sum(e)
    # Gradient: dE/dr_i = 4 * sum_j (-12 r^-14 + 6 r^-7) * (r_i - r_j) / r
    # = 24 * sum_j (2 r^-14 - r^-7) * (r_i - r_j)
    # In vectorized form: grad_i = 24 * sum_j (2*r12 - r6) / r2 * diff[i,j]
    # Actually: force on i from j = 24*(2/r^14 - 1/r^7)*(r_i - r_j)
    # grad[i] = sum_j 24*(2*r12[i,j] - r6[i,j])/r2[i,j] * diff[i,j]
    # Let's compute properly:
    # dE/dpos_i = 4 * sum_{j != i} d/dr (r^-12 - r^-6) * d r / d pos_i
    #           = 4 * sum_j (-12 r^-13 + 6 r^-7) * (pos_i - pos_j)/r
    #           = 4 * sum_j (-12 r^-14 + 6 r^-8) * (pos_i - pos_j)
    #           = 24 * sum_j (-2 r^-14 + r^-8) * (pos_i - pos_j)
    #           = 24 * sum_j (r^-8 - 2 r^-14) * (pos_i - pos_j)
    # With r2, r6, r12: r^-8 = r6/r2, r^-14 = r12/r2
    coeff = 24.0 * (r6 / r2 - 2.0 * r12 / r2)
    coeff[0, 0] = 0.0
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _fcc_truncated_octahedron(scale=1.0, strains=None):
    """
    Generate FCC truncated octahedron with 38 atoms.
    The FCC truncated octahedron for n=38 has shells:
    - Shell 1: 1 atom at center
    - Shell 2: 12 atoms (FCC nearest neighbors at distance a)
    - Shell 3: 6 atoms (face centers, distance a*sqrt(2))
    - Shell 4: 8 atoms (edge midpoints, distance a*sqrt(3))
    - Shell 5: 11 atoms (next shell)
    
    Actually, the standard LJ38 global minimum is a truncated octahedron with:
    1 + 12 + 6 + 8 + 11 = 38? Let me use the known coordinates.
    
    The FCC truncated octahedron has 38 atoms arranged as:
    - 1 at origin
    - 12 at (±1, ±1, 0) type positions (FCC neighbors)
    - 6 at (±2, 0, 0) type
    - 8 at (±1, ±1, ±1) type
    - 11 more...
    
    Let me use the standard construction: take FCC lattice points within a truncated octahedron radius.
    """
    # Standard FCC basis vectors
    # FCC lattice: points at (i, j, k) where i+j+k is even, scaled by a/sqrt(2)
    # Or equivalently, the conventional FCC cell has atoms at corners and face centers.
    
    # Simpler: generate FCC points and select those within truncated octahedron
    # Truncated octahedron: |x|+|y|+|z| <= a and max(|x|,|y|,|z|) <= b
    
    # Known LJ38 global minimum structure (from literature):
    # It's a truncated octahedron with FCC symmetry.
    # Coordinates can be generated from FCC lattice.
    
    # FCC lattice points: all (i, j, k) with i+j+k even, in units of a/2
    # where a is the nearest-neighbor distance.
    
    # Let's generate a sufficiently large FCC lattice and clip to truncated octahedron
    # Truncated octahedron in FCC: the 38-atom cluster corresponds to
    # all FCC lattice points (i,j,k) with i+j+k even such that
    # the point is inside the Wigner-Seitz cell of the truncated octahedron.
    
    # Actually, the simplest correct construction:
    # The 38-atom FCC truncated octahedron consists of:
    # Layer 0 (z=0): 1 atom at (0,0,0)
    # Layer 1 (z=1): 12 atoms
    # Layer 2 (z=2): 6 atoms  
    # Layer 3 (z=3): 8 atoms
    # Layer 4 (z=4): 11 atoms
    # Total: 38
    
    # Let me use explicit coordinates from the known structure.
    # In FCC units (nearest neighbor = 1), the 38 atoms are:
    
    # Using the standard reference: the 38-atom truncated octahedron
    # has the following FCC lattice points (in units where nn distance = sqrt(2)/2 * a_conventional)
    
    # I'll use a direct construction based on the known energy -173.928427
    
    # Generate FCC points: (i, j, k) where i+j+k is even
    # Scale so that nearest neighbor distance = 1.12 (LJ minimum)
    
    # FCC lattice vectors: a1 = (0.5, 0.5, 0), a2 = (0.5, 0, 0.5), a3 = (0, 0.5, 0.5)
    # Lattice point: n1*a1 + n2*a2 + n3*a3 = (0.5*(n1+n2), 0.5*(n1+n3), 0.5*(n2+n3))
    # Nearest neighbor distance = 0.5*sqrt(2) = 0.7071...
    
    # To get nn distance = d, scale by d/0.7071 = d*sqrt(2)
    
    # For the truncated octahedron with 38 atoms, we need points with
    # |n1|, |n2|, |n3| <= some cutoff
    
    # Let me just enumerate FCC points and select the 38 closest to origin
    # that form a truncated octahedron shape.
    
    # Known result: the 38-atom truncated octahedron contains all FCC points
    # (n1, n2, n3) with max(|n1|,|n2|,|n3|) <= 2 and n1+n2+n3 even
    # Let's count: for max coord <= 2, even sum:
    # This gives a specific set. Let me compute it.
    
    points = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                if (n1 + n2 + n3) % 2 == 0:
                    # FCC lattice point
                    x = 0.5 * (n1 + n2)
                    y = 0.5 * (n1 + n3)
                    z = 0.5 * (n2 + n3)
                    r = math.sqrt(x*x + y*y + z*z)
                    points.append((r, x, y, z))
    
    points.sort(key=lambda p: p[0])
    
    # Take the 38 closest
    selected = points[:N]
    coords = np.array([[x, y, z] for (_, x, y, z) in selected], dtype=float)
    
    # Scale to LJ minimum distance
    # Nearest neighbor in FCC is at distance 0.5*sqrt(2) ≈ 0.7071
    # We want nn distance ≈ 1.12
    nn_dist = 0.5 * math.sqrt(2)
    scale_factor = 1.12 / nn_dist
    coords = coords * scale_factor
    
    # Apply strains if provided (3x3 symmetric matrix for local deformation)
    if strains is not None:
        coords = coords @ strains.T
    
    return coords


def _basin_hopping(pos0, n_iters=200, T_start=1.0, T_end=0.01, step_size=0.1):
    """Simple basin hopping with L-BFGS relaxation."""
    from scipy.optimize import minimize
    
    best_pos = pos0.copy()
    best_e, _ = _lj_energy_and_grad(best_pos)
    
    T = T_start
    for it in range(n_iters):
        if time.time() > _DEADLINE:
            break
        
        # Random perturbation
        noise = np.random.randn(*pos0.shape) * step_size
        perturbed = pos0 + noise
        
        # Check minimum distance constraint
        diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        if np.min(r2) < 0.25:  # 0.5^2
            continue
        
        # Local relaxation
        try:
            result = minimize(
                lambda x: _lj_energy_and_grad(x)[0],
                perturbed,
                jac=lambda x: _lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_pos = result.x
            new_e, _ = _lj_energy_and_grad(new_pos)
            
            # Check validity
            diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, 1.0)
            if np.min(r2) < 0.25:
                continue
            
            # Metropolis acceptance
            delta_e = new_e - best_e
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / T):
                best_pos = new_pos.copy()
                best_e = new_e
                pos0 = new_pos.copy()
        except Exception:
            pass
        
        # Anneal
        T = T_start * (T_end / T_start) ** (it / max(n_iters - 1, 1))
    
    return best_pos, best_e


def construct():
    np.random.seed(42)
    
    # Start with FCC truncated octahedron
    pos = _fcc_truncated_octahedron()
    e, _ = _lj_energy_and_grad(pos)
    
    # First, do a local relaxation to polish the initial structure
    from scipy.optimize import minimize
    try:
        result = minimize(
            lambda x: _lj_energy_and_grad(x)[0],
            pos,
            jac=lambda x: _lj_energy_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pos = result.x
        e, _ = _lj_energy_and_grad(pos)
    except Exception:
        pass
    
    best_pos = pos.copy()
    best_e = e
    
    # Basin hopping from the relaxed FCC structure
    if time.time() < _DEADLINE - 10:
        bh_pos, bh_e = _basin_hopping(best_pos, n_iters=150, T_start=0.5, T_end=0.01, step_size=0.08)
        if bh_e < best_e:
            best_pos = bh_pos
            best_e = bh_e
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            result = minimize(
                lambda x: _lj_energy_and_grad(x)[0],
                best_pos,
                jac=lambda x: _lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            final_pos = result.x
            final_e, _ = _lj_energy_and_grad(final_pos)
            
            # Validate minimum distance
            diff = final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, 1.0)
            if np.min(r2) >= 0.25 and final_e < best_e:
                best_pos = final_pos
                best_e = final_e
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
