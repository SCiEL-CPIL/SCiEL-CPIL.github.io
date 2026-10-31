"""LJ38 global minimum search: FCC truncated octahedron + local relaxation + multi-start."""
import numpy as np
import time
import sys

N = 38

# Deadline guard: evaluator kills at 150s, we return well before that.
_DEADLINE = 140.0  # seconds from start of construct()
_t0 = None

def _lj_energy(pos):
    """Compute total LJ energy: 4 * sum_{i<j} (1/r^12 - 1/r^6)."""
    n = len(pos)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-12:
                return 1e10
            inv_r2 = 1.0 / (r * r)
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 * inv_r6
            total += 4.0 * (inv_r12 - inv_r6)
    return total

def _lj_energy_vec(pos):
    """Vectorized LJ energy computation."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Filter out zeros (shouldn't happen but safety)
    dist = dist[dist > 1e-12]
    inv_r2 = 1.0 / (dist * dist)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6
    # Each pair counted twice, so divide by 2
    return 2.0 * np.sum(4.0 * (inv_r12 - inv_r6)) / 2.0

def _fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron cluster of 38 atoms.
    The FCC truncated octahedron has:
    - 8 corner atoms (cubic corners)
    - 12 face-center atoms
    - 12 edge-center atoms  
    - 6 additional atoms (or the standard 38-atom FCC truncation)
    
    Actually, the standard 38-atom FCC cluster is a truncated octahedron with:
    - 1 central atom
    - 12 first-shell neighbors (octahedral + tetrahedral)
    - 15 second-shell
    - 10 third-shell
    Total: 1 + 12 + 15 + 10 = 38
    
    Let me use the known coordinates for the 38-atom FCC truncated octahedron.
    The FCC lattice points within a truncated octahedron of appropriate size.
    
    FCC lattice: points at (i, j, k) where i+j+k is even, scaled by a/2.
    For the 38-atom cluster, we need the appropriate truncation.
    
    Let me construct it properly. The 38-atom FCC cluster has the structure:
    Layer 1 (center): 1 atom at (0,0,0)
    Layer 2: 12 atoms at FCC nearest neighbors
    Layer 3: 15 atoms
    Layer 4: 10 atoms
    
    Actually, let me just use the well-known construction. The 38-atom FCC 
    truncated octahedron can be built from FCC lattice points.
    
    FCC lattice points: (a/2)(i, j, k) where i+j+k is even.
    For the 38-atom cluster, we take all FCC points within a certain radius.
    
    Let me enumerate FCC points and select the 38 closest to origin.
    """
    a = 1.0  # lattice parameter (will be scaled)
    # FCC points: (a/2)*(i,j,k) with i+j+k even
    # Enumerate a range of indices
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    pts.append((a/2 * i, a/2 * j, a/2 * k))
    
    # Sort by distance from origin
    pts = sorted(pts, key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    # Take the 38 closest (including origin)
    selected = pts[:N]
    return np.array(selected, dtype=float)

def _try_relax(pos, max_time=10.0):
    """Attempt local relaxation using L-BFGS if available, otherwise simple gradient descent."""
    global _t0
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback: simple gradient descent
        pos = pos.copy()
        lr = 0.01
        for it in range(200):
            if _t0 is not None and (time.time() - _t0) > _DEADLINE - 5:
                break
            # Compute gradient numerically
            n = len(pos)
            grad = np.zeros_like(pos)
            for i in range(n):
                for d in range(3):
                    eps = 1e-6
                    pos[i, d] += eps
                    e_plus = _lj_energy(pos)
                    pos[i, d] -= 2 * eps
                    e_minus = _lj_energy(pos)
                    pos[i, d] += eps
                    grad[i, d] = (e_plus - e_minus) / (2 * eps)
            pos -= lr * grad
            # Check convergence
            if np.max(np.abs(grad)) < 1e-8:
                break
        return pos
    
    def energy_and_grad(x):
        pos = x.reshape(-1, 3)
        n = len(pos)
        e = 0.0
        grad = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[i] - pos[j]
                r2 = np.dot(r_vec, r_vec)
                if r2 < 1e-12:
                    return 1e10, grad
                r = np.sqrt(r2)
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3
                inv_r12 = inv_r6 * inv_r6
                e += 4.0 * (inv_r12 - inv_r6)
                # dE/dr = 4*(-12/r^13 + 6/r^7)
                # dE/dx_i = -dE/dr * r_vec/r = (48*inv_r12 - 24*inv_r6) * r_vec / r
                # = (48*inv_r12 - 24*inv_r6) * r_vec * inv_r
                coeff = (48.0 * inv_r12 - 24.0 * inv_r6) / r
                grad[i] += coeff * r_vec
                grad[j] -= coeff * r_vec
        return e, grad

    x0 = pos.flatten()
    
    # Use L-BFGS-B with time limit
    best_x = x0.copy()
    best_e = energy_and_grad(x0)[0]
    
    start_time = time.time()
    try:
        result = minimize(
            energy_and_grad, x0, method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-10}
        )
        if result.success or result.fun < best_e:
            best_x = result.x
            best_e = result.fun
    except Exception:
        pass
    
    # Check deadline
    if _t0 is not None and (time.time() - _t0) > _DEADLINE - 5:
        return best_x.reshape(-1, 3)
    
    return best_x.reshape(-1, 3)

def construct():
    global _t0
    _t0 = time.time()
    
    best_pos = None
    best_energy = 1e10
    
    # Strategy 1: Start from FCC truncated octahedron
    fcc_pos = _fcc_truncated_octahedron()
    
    # Relax the FCC structure
    relaxed = _try_relax(fcc_pos)
    e = _lj_energy(relaxed)
    if e < best_energy:
        best_energy = e
        best_pos = relaxed.copy()
    
    # Check deadline
    if (time.time() - _t0) > _DEADLINE - 20:
        return best_pos
    
    # Strategy 2: Multi-start with small perturbations around FCC
    rng = np.random.RandomState(42)
    n_starts = 10
    for start_idx in range(n_starts):
        if (time.time() - _t0) > _DEADLINE - 15:
            break
        
        # Add small random perturbation
        perturbation = rng.randn(N, 3) * 0.1
        perturbed = fcc_pos + perturbation
        
        # Relax
        relaxed = _try_relax(perturbed)
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Strategy 3: Try different scaling of FCC
    if (time.time() - _t0) > _DEADLINE - 10:
        return best_pos
    
    for scale in [0.9, 0.95, 1.05, 1.1]:
        if (time.time() - _t0) > _DEADLINE - 10:
            break
        scaled = fcc_pos * scale
        relaxed = _try_relax(scaled)
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Final polish: tighter relaxation on best result
    if best_pos is not None and (time.time() - _t0) < _DEADLINE - 5:
        polished = _try_relax(best_pos)
        e = _lj_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pos = polished.copy()
    
    return best_pos

# Ensure construct is the entry point
if __name__ == "__main__":
    pos = construct()
    print(f"Energy: {_lj_energy(pos)}")
    print(f"Shape: {pos.shape}")
