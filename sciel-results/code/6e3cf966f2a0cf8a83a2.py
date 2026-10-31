"""Two-stage topology-forcing strategy for LJ38 global minimum."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

# Known FCC truncated octahedron shell radii (reduced units, sigma=1)
# The global minimum has energy -173.928427
# Shell structure: 1 (center) + 12 (1st shell) + 6 (2nd) + 12 (3rd) + 7 (4th)
# Approximate radii for FCC truncated octahedron:
# Center: 0
# 1st shell (12 atoms): ~1.0
# 2nd shell (6 atoms): ~1.414
# 3rd shell (12 atoms): ~1.732
# 4th shell (7 atoms): ~2.0-2.2

def lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * n * (1.0 - 1.0)  # subtract self terms (zero anyway)
    # More carefully:
    mask = ~np.eye(n, dtype=bool)
    energy = 4.0 * np.sum((r12 - r6)[mask])
    return energy

def lj_forces(pos):
    """Compute Lennard-Jones forces (negative gradient of energy)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    # Force on atom i from atom j: 4 * (12*r12 - 6*r6) / r2 * diff_ij
    # F = -dU/dr = 4*(12*r^-14 - 6*r^-8) * r_vec
    # In terms of r2: F = 4*(12*r12 - 6*r6)/r2 * diff
    coeff = 4.0 * (12.0 * r12 - 6.0 * r6) / r2
    np.fill_diagonal(coeff, 0.0)
    forces = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces

def shell_radii_loss(pos, shell_targets):
    """
    Smooth loss that penalizes deviation from expected shell radii.
    shell_targets: list of (radius, count) tuples
    """
    n = len(pos)
    # Compute distances from center of mass
    com = np.mean(pos, axis=0)
    rel = pos - com
    dists = np.sqrt(np.sum(rel**2, axis=1))
    
    loss = 0.0
    # Sort distances to assign to shells
    sorted_dists = np.sort(dists)
    
    idx = 0
    for radius, count in shell_targets:
        shell_dists = sorted_dists[idx:idx+count]
        # Smooth penalty: mean squared deviation from target radius
        loss += np.mean((shell_dists - radius)**2)
        idx += count
    
    return loss

def topology_loss(pos):
    """
    Combined topology loss: penalize deviation from FCC truncated octahedron
    shell structure.
    """
    # FCC truncated octahedron shell structure for 38 atoms:
    # Shell 0: 1 atom at center (radius 0)
    # Shell 1: 12 atoms (radius ~1.0)
    # Shell 2: 6 atoms (radius ~1.414)
    # Shell 3: 12 atoms (radius ~1.732)
    # Shell 4: 7 atoms (radius ~2.05)
    shell_targets = [
        (0.0, 1),
        (1.0, 12),
        (1.414, 6),
        (1.732, 12),
        (2.05, 7),
    ]
    
    return shell_radii_loss(pos, shell_targets)

def combined_objective(pos, alpha=1.0):
    """
    Combined objective: alpha * topology_loss + (1-alpha) * normalized LJ energy
    This is a smooth surrogate that guides toward the FCC structure.
    """
    topo = topology_loss(pos)
    energy = lj_energy(pos)
    # Normalize energy to be comparable to topology loss
    # Typical energy range: -170 to -100
    norm_energy = energy / 200.0
    return alpha * topo + (1.0 - alpha) * norm_energy

def gradient_combined(pos, alpha=1.0):
    """
    Analytical gradient of the combined objective.
    """
    n = len(pos)
    com = np.mean(pos, axis=0)
    rel = pos - com
    dists = np.sqrt(np.sum(rel**2, axis=1) + 1e-12)
    
    # Gradient of topology loss w.r.t. positions
    # d/dpos_i [mean((d_j - r_target)^2) for j in shell]
    # = (2/d_j) * (d_j - r_target) * (pos_i - com) / count_shell
    # Simplified: use numerical gradient for topology, analytical for LJ
    
    topo_grad = np.zeros_like(pos)
    sorted_dists, sort_idx = np.sort(dists), np.argsort(dists)
    
    shell_targets = [
        (0.0, 1),
        (1.0, 12),
        (1.414, 6),
        (1.732, 12),
        (2.05, 7),
    ]
    
    idx = 0
    for radius, count in shell_targets:
        shell_indices = sort_idx[idx:idx+count]
        for i in shell_indices:
            d = dists[i]
            if d < 1e-6:
                continue
            # Gradient contribution
            topo_grad[i] += (2.0 * (d - radius) / count) * rel[i] / d
        idx += count
    
    # Gradient of LJ energy
    lj_grad = -lj_forces(pos)  # gradient = -forces
    
    return alpha * topo_grad + (1.0 - alpha) * lj_grad / 200.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Stage 1: Topology-forcing search
    # Start from multiple random initializations biased toward FCC-like structure
    best_energy = np.inf
    best_pos = None
    
    np.random.seed(42)
    
    # Generate initial positions near FCC truncated octahedron
    # Center atom
    pos0 = np.zeros((1, 3))
    
    # 1st shell: 12 atoms at vertices of cuboctahedron (FCC neighbors)
    # FCC neighbors are at (±1,±1,0), (±1,0,±1), (0,±1,±1) scaled by 1/sqrt(2)
    fcc_dir = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                if abs(sx) + abs(sy) + abs(sz) == 2:
                    fcc_dir.append([sx, sy, sz])
    fcc_dir = np.array(fcc_dir, dtype=float)
    fcc_dir /= np.sqrt(2.0)
    shell1 = fcc_dir * 1.0
    
    # 2nd shell: 6 atoms at (±1,0,0), (0,±1,0), (0,0,±1) * sqrt(2)
    shell2_dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    shell2 = shell2_dirs * 1.414
    
    # 3rd shell: 12 atoms - use FCC lattice points at distance ~1.732
    # These are at (±1,±1,±1) normalized * 1.732
    shell3_dirs = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                shell3_dirs.append([sx, sy, sz])
    shell3_dirs = np.array(shell3_dirs, dtype=float)
    shell3_dirs /= np.sqrt(3.0)
    shell3 = shell3_dirs * 1.732
    
    # 4th shell: 7 atoms - approximate positions
    # Use some FCC lattice points at larger distance
    shell4 = np.array([
        [2.0, 0, 0], [-2.0, 0, 0],
        [0, 2.0, 0], [0, -2.0, 0],
        [0, 0, 2.0], [0, 0, -2.0],
        [1.0, 1.0, 1.0] * 0.5  # 7th atom
    ])
    # Adjust 7th atom to be distinct
    shell4[6] = [0.5, 0.5, 1.8]
    
    base_pos = np.vstack([pos0, shell1, shell2, shell3, shell4])
    
    # Run multiple restarts with perturbations
    n_restarts = 30
    results = []
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb the base structure
        if restart == 0:
            pos = base_pos.copy()
        else:
            # Add random perturbation
            noise_scale = 0.1 + 0.3 * np.random.random()
            pos = base_pos + noise_scale * np.random.randn(N, 3)
        
        # Stage 1a: Minimize topology loss with small alpha
        try:
            res1 = minimize(
                combined_objective,
                pos,
                args=(0.8,),
                jac=lambda p, a: gradient_combined(p, a),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if time.time() > deadline - 20.0:
                break
        except Exception:
            continue
        
        pos1 = res1.x
        e1 = lj_energy(pos1)
        
        # Stage 1b: Minimize LJ energy from the topology-guided position
        try:
            res2 = minimize(
                lj_energy,
                pos1,
                jac=lambda p: -lj_forces(p),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if time.time() > deadline - 10.0:
                break
        except Exception:
            continue
        
        pos2 = res2.x
        e2 = lj_energy(pos2)
        
        # Check minimum distance constraint
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos2[i] - pos2[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist >= 0.5 and e2 < best_energy:
            best_energy = e2
            best_pos = pos2.copy()
        
        results.append(e2)
    
    # If no valid result, fall back to seed
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = lj_energy(best_pos)
    
    # Final polish: high-precision local relaxation
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                lj_energy,
                best_pos,
                jac=lambda p: -lj_forces(p),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            pos_final = res_final.x
            e_final = lj_energy(pos_final)
            
            # Verify minimum distance
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos_final[i] - pos_final[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist >= 0.5 and e_final < best_energy:
                best_pos = pos_final
                best_energy = e_final
        except Exception:
            pass
    
    return best_pos
