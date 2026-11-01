"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _lj_energy_and_grad(pos):
    """
    Compute the LJ energy and gradient for N atoms.
    Energy: 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    """
    pos = pos.reshape(N, 3)
    # Compute all pair differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=-1)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, np.inf)
    
    # Energy: sum over i<j, which is half the sum over i!=j
    # 4 * sum_{i<j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 2.0 * np.sum(1.0 / r12 - 1.0 / r6)
    
    # Gradient
    # dE/dx_i = sum_{j != i} 4 * (6 r_ij^-8 - 12 r_ij^-14) * (x_i - x_j)
    # Note: r^-8 = (r^2)^-4, r^-14 = (r^2)^-7
    # factor = 4 * (6 * r2^-4 - 12 * r2^-7)
    factor = 4.0 * (6.0 * r2 ** (-4) - 12.0 * r2 ** (-7))
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _make_mackay_icosahedron():
    """
    Generate the 55-atom Mackay icosahedron geometry.
    Structure:
    - 1 central atom
    - 12 vertices of an icosahedron (radius R1)
    - 32 atoms in the second shell:
        - 12 face centers (radius R2)
        - 20 edge midpoints (radius R3)
    Total: 1 + 12 + 12 + 20 = 45. Wait, 55 atoms.
    Let's re-examine the Mackay icosahedron for N=55.
    The standard Mackay icosahedra sizes are 1, 13, 55, 147, ...
    N = (3n^3 + 5n^2 + 2n) / 2
    n=1: 5 (triangular bipyramid? No, 5 is a square pyramid or trigonal bipyramid)
    n=2: 24? No.
    Let's check the formula again.
    Mackay icosahedra:
    n=1: 13 atoms (1 center + 12 vertices)
    n=2: 55 atoms (1 center + 12 vertices + 32 in 2nd shell)
    The 32 atoms in the 2nd shell consist of:
    - 12 atoms at the face centers of the icosahedron
    - 20 atoms at the edge midpoints of the icosahedron
    12 + 20 = 32.
    So 1 + 12 + 32 = 45. This is not 55.
    
    Wait, the 55-atom Mackay icosahedron has a different composition.
    Let's look up the structure.
    The 55-atom Mackay icosahedron is composed of 13 icosahedra of 5 atoms each? No.
    It is a closed-shell structure.
    Actually, the 55-atom Mackay icosahedron has:
    - 1 central atom
    - 12 atoms at the vertices of a regular icosahedron
    - 32 atoms at the face centers and edge midpoints?
    
    Let's use a known coordinate set or generate it properly.
    The 55-atom Mackay icosahedron can be constructed by taking the 13-atom icosahedron
    and adding a second shell.
    
    Alternatively, I can use the fact that the global minimum is known to be the Mackay icosahedron
    and use a relaxation from a reasonable starting point.
    
    Let's try to construct the 55-atom Mackay icosahedron properly.
    The icosahedron has 12 vertices, 30 edges, and 20 faces.
    The 13-atom icosahedron: 1 center + 12 vertices.
    The 55-atom Mackay icosahedron:
    - 1 center
    - 12 vertices of the icosahedron
    - 32 atoms in the second shell:
        - 12 atoms at the face centers
        - 20 atoms at the edge midpoints
    This gives 1 + 12 + 12 + 20 = 45 atoms. Still not 55.
    
    I must be missing something. Let's check the Cambridge Cluster Database value.
    Energy -279.248470 for N=55.
    
    Let's try a different approach. I'll use the 13-atom icosahedron as a seed
    and add atoms in a way that mimics the Mackay growth.
    
    Actually, let's just use a cubic lattice fragment with the correct spacing
    and relax it. The cubic lattice fragment is a valid starting point.
    """
    # Generate icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Scale to the correct radius for the 13-atom icosahedron
    # The equilibrium distance in LJ is 2^(1/6) ≈ 1.12246
    # The distance from center to vertex in the 13-atom icosahedron is r
    # The distance between adjacent vertices is 2 * r * sin(pi/5)
    # We want the nearest-neighbor distance to be 2^(1/6)
    # sin(pi/5) = sin(36 degrees) ≈ 0.5878
    # 2 * r * 0.5878 = 1.12246
    # r = 1.12246 / (2 * 0.5878) ≈ 0.954
    r_vertex = (2.0 ** (1.0 / 6.0)) / (2.0 * math.sin(math.pi / 5.0))
    vertices = vertices * r_vertex
    
    # 13-atom icosahedron: center + 12 vertices
    pos = np.vstack([np.zeros(3), vertices])
    
    # Now add 42 more atoms to reach 55
    # This is where it gets complicated. Let's just use a cubic lattice fragment
    # as the starting point for relaxation.
    
    # Generate cubic lattice fragment
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    # Sort by distance from center
    center = (side - 1) / 2.0
    points.sort(key=lambda p: (p[0] - center) ** 2 + (p[1] - center) ** 2 + (p[2] - center) ** 2)
    init_pos = np.array(points[:N], dtype=float) * (2.0 ** (1.0 / 6.0))
    
    # Center the structure
    init_pos -= init_pos.mean(axis=0)
    
    return init_pos


def construct():
    """
    Construct a low-energy configuration for 55 Lennard-Jones atoms.
    Strategy:
    1. Start from a cubic lattice fragment with correct spacing.
    2. Perform local relaxation using L-BFGS to find a local minimum.
    3. Try multiple random perturbations to escape local minima.
    4. Return the best configuration found.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for evaluator overhead
    
    # Initial configuration
    init_pos = _make_mackay_icosahedron()
    
    # Define the energy function
    def energy_func(x):
        return _lj_energy_and_grad(x)[0]
    
    # Define the gradient function
    def grad_func(x):
        return _lj_energy_and_grad(x)[1]
    
    # Relax the initial configuration
    try:
        result = minimize(
            fun=energy_func,
            x0=init_pos.flatten(),
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        best_pos = result.x.reshape(N, 3)
        best_energy = result.fun
    except Exception:
        best_pos = init_pos
        best_energy = energy_func(init_pos.flatten())
    
    # Check time
    if time.time() > deadline - 20.0:
        return best_pos
    
    # Try multiple random perturbations
    rng = np.random.RandomState(42)
    for i in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the best configuration
        perturbed = best_pos + rng.normal(0, 0.1, (N, 3))
        
        try:
            result = minimize(
                fun=energy_func,
                x0=perturbed.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_energy:
                best_pos = result.x.reshape(N, 3)
                best_energy = result.fun
        except Exception:
            continue
    
    return best_pos
# EVOLVE-BLOCK-END
