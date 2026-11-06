"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

# Global start time to track budget across calls if needed, but task says "called once for each... all six calls together must finish within 300s".
# It's safer to assume a per-call budget or a global one. The prompt says "Each evaluation must finish within 300 s" in the end, but the contract says "all six calls together must finish within 300 s".
# Usually, these evaluators run the function 6 times sequentially. Let's track a global deadline.
_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Safety margin

def _get_remaining_time():
    return max(0.0, _TOTAL_BUDGET - (time.time() - _START_TIME))

def _generate_fcc(n, rng, scale=1.0):
    """Generate an FCC lattice fragment."""
    points = []
    a = scale
    # Generate a large enough grid
    L = int(np.ceil(n ** (1/3))) + 2
    for i in range(-L, L):
        for j in range(-L, L):
            for k in range(-L, L):
                # FCC basis vectors
                base = np.array([i*a, j*a, k*a])
                for b in [np.array([0,0,0]), np.array([a/2, a/2, 0]), np.array([a/2, 0, a/2]), np.array([0, a/2, a/2])]:
                    p = base + b
                    if len(points) < n:
                        # Add small noise to break symmetry
                        p += rng.normal(0, 0.01, 3)
                        points.append(p)
    return np.array(points[:n])

def _generate_hcp(n, rng, scale=1.0):
    """Generate an HCP lattice fragment."""
    points = []
    a = scale
    c = a * np.sqrt(8/3)
    L = int(np.ceil(n ** (1/3))) + 2
    for i in range(-L, L):
        for j in range(-L, L):
            for k in range(-L, L):
                # HCP structure: ABAB stacking
                # Layer 0 (A): (0,0,0)
                # Layer 1 (B): (1/3, 1/sqrt(3), 1/2) * a? No, let's use standard coords.
                # Simpler: Use FCC and shift some layers? Or just generate points in hexagonal grid.
                # Let's stick to a simpler heuristic: Random points in a cylinder with hexagonal packing bias?
                # Actually, let's just use a dense random sphere for variety, FCC is a good structured one.
                # HCP is tricky to generate a fragment of exactly N without complex logic.
                # Let's use a spherical shell instead for the second structured type.
                pass
    return np.array(points[:n])

def _generate_spherical_shell(n, rng, radius=1.0):
    """Generate points on a spherical shell (Fibonacci sphere)."""
    # Fibonacci sphere for even distribution
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    z = 1 - (i + 0.5) * (2.0 / n)
    r = np.sqrt(1 - z**2)
    theta = 2 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, z)) * radius
    # Add noise
    points += rng.normal(0, 0.05, 3)
    return points

def _generate_random(n, rng, spacing=1.0):
    """Generate random compact points."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points (slow for large n, but n<=80)
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
    return np.array(points)

def construct(instance):
    """Best of diverse starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42 + size) # Deterministic seed per size for reproducibility in testing? 
    # Better to use a global rng or fixed seed to ensure consistency? 
    # The evaluator might run multiple times? "called once for each". 
    # Let's use a seed based on instance to be deterministic.
    
    best_energy = np.inf
    best_positions = None
    
    # List of initial generators
    generators = [
        lambda: _generate_fcc(size, rng, scale=1.0),
        lambda: _generate_spherical_shell(size, rng, radius=1.5 * size**(1/3)),
        lambda: _generate_random(size, rng, spacing=1.0),
        lambda: _generate_random(size, rng, spacing=1.2), # Looser packing
    ]
    
    # We can also try a "vibrational" start: take the best so far, perturb, relax.
    # But let's start with the static ones.
    
    # Allocate time per attempt. 
    # Total budget ~290s. 6 instances. ~48s per instance.
    # Let's run a fixed number of attempts if time permits.
    
    # First pass: Run the 4 generators
    for gen in generators:
        if _get_remaining_time() < 5.0: break
        try:
            points = gen()
            # Ensure minimum distance constraint is roughly met or let relax fix it?
            # Relax might fail if points are too close.
            # The task says "atoms must be at least 0.5 apart".
            # If relax returns a valid structure, it should satisfy this.
            energy, positions = relax(points, potential, parameter)
            # Check validity: min distance
            if len(positions) > 0:
                # Simple check: if any pair < 0.5, it's invalid.
                # Relax usually handles this, but let's be safe.
                if np.all(np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1) > 0.5) or np.min(np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1)) >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
        except Exception:
            continue

    # Second pass: Basin hopping / Perturbation on the best found
    # If we have time, perturb the best solution and relax again.
    if best_positions is not None:
        perturbations = [0.05, 0.1, 0.2, 0.5]
        for amp in perturbations:
            if _get_remaining_time() < 5.0: break
            try:
                # Create a few perturbed copies
                for _ in range(2):
                    if _get_remaining_time() < 2.0: break
                    perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                    energy, positions = relax(perturbed, potential, parameter)
                    if len(positions) > 0:
                        min_dist = np.min(np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1))
                        if min_dist >= 0.5 - 1e-6:
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = positions
            except Exception:
                continue

    # If we still have time, we can try more random starts
    while _get_remaining_time() > 10.0:
        try:
            points = _generate_random(size, rng, spacing=1.0)
            energy, positions = relax(points, potential, parameter)
            if len(positions) > 0:
                min_dist = np.min(np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1))
                if min_dist >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
        except Exception:
            break

    if best_positions is None:
        # Fallback: return a valid random structure
        points = _generate_random(size, rng, spacing=1.0)
        energy, positions = relax(points, potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
