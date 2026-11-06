"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _local_relax(pos, potential, parameter):
    """
    Use L-BFGS-B with gradient to find the local minimum.
    Returns (energy, position).
    """
    try:
        # L-BFGS-B is efficient for small to medium sized problems with gradients
        # Note: We assume dzugutov_energy_gradient returns (energy, gradient) as is common in such tasks,
        # or we need to wrap it. Given the name, it's ambiguous. 
        # However, standard scipy minimize with jac=None expects the function to return f.
        # If the function returns (f, g), we can pass it directly if it's structured correctly,
        # but usually we need to separate them or use a wrapper.
        # Let's assume the function returns (energy, gradient) tuple.
        # scipy minimize can accept a function that returns (f, g) if we don't specify jac? No, 
        # if jac is not specified, it uses finite differences on the returned value.
        # If the function returns a tuple, that will fail.
        # We need to create a wrapper that returns only energy, and another for gradient,
        # OR if the function returns (f, g), we can pass a lambda that returns f and use jac=lambda x: g.
        
        def energy_func(x):
            result = dzugutov_energy_gradient(x)
            return result[0]

        def gradient_func(x):
            result = dzugutov_energy_gradient(x)
            return result[1]

        res = minimize(
            energy_func,
            pos,
            jac=gradient_func,
            method='L-BFGS-B',
            options={'disp': False}
        )
                                      # and you have to compute energy separately or the function returns a tuple.
                                      # Let's assume it returns the gradient only? 
                                      # No, usually `relax` is the safe bet. 
                                      # But wait, if `dzugutov_energy_gradient` returns the gradient, 
                                      # we need the energy function too. 
                                      # The task says "Energies are recomputed exactly...". 
                                      # It implies we can compute E.
                                      # Let's look at the current code: it uses `relax(pos, potential, parameter)`.
                                      # If `dzugutov_energy_gradient` is just the gradient, we can't use it directly with minimize 
                                      # unless we have an energy function.
                                      # Let's assume `dzugutov_energy_gradient` might return (energy, gradient) or just gradient.
                                      # To be safe and robust, I will stick to `relax` for the local step if I'm unsure about the return type of the gradient function, 
                                      # BUT the prompt says "dzugutov_energy_gradient(positions) may be called", implying it's a useful tool.
                                      # Often in these problems, `f` and `jac` are separate.
                                      # Let's try to use `relax` but maybe wrap it in a more robust search.
                                      # Actually, let's look at the current score: 3.4086. 
                                      # If `relax` is slow, that's a bottleneck.
                                      # Let's assume `dzugutov_energy_gradient` returns the gradient vector.
                                      # We need an energy function. The task doesn't provide one explicitly in the interface list, 
                                      # but `relax` returns energy.
                                      # Let's define a wrapper that uses `relax` if we can't use the gradient directly, 
                                      # OR assume `dzugutov_energy_gradient` returns (E, G)? 
                                      # Let's check common patterns. Usually, it's `energy` and `gradient`.
                                      # If I have to guess, using `relax` is safer.
                                      # However, the directive says "design a new program... from scratch".
                                      # Let's use a **Simulated Annealing** approach with `relax` as the cooling step.
                                      # This is distinct from Basin Hopping (which is greedy/Metropolis on kicks).
                                      # SA has a temperature schedule.
    except Exception as e:
        return float('inf'), pos

    # Fallback to relax if gradient usage is ambiguous or fails
    try:
        energy, pos_relaxed = relax(pos, potential, parameter)
        return energy, pos_relaxed
    except Exception:
        return float('inf'), pos

def _generate_random_shell(n, rng, r_min=1.0, r_max=2.0):
    """Generate n points randomly distributed in a spherical shell."""
    pts = np.zeros((n, 3))
    for i in range(n):
        # Uniform distribution in volume of shell
        r = rng.uniform(r_min ** 0.5, r_max ** 0.5) ** 2 # Actually uniform in r^3? No, uniform in volume is r = (r_min^3 + rng*(r_max^3-r_min^3))^(1/3)
        r = (r_min**3 + rng.random() * (r_max**3 - r_min**3)) ** (1/3)
        # Random direction
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts[i] = v * r
    return pts

def _generate_icosahedral(n, rng):
    """
    Generate n points arranged in concentric icosahedral shells.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    
    # Shell 0: Center
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        n -= 1
    
    # Shell 1: 12 vertices
    if n > 0:
        shell_count = min(12, n)
        for i in range(shell_count):
            r = 1.1
            p = verts[i] * r
            points.append(p)
        n -= shell_count
    
    # Shell 2: 42 vertices (approximate)
    # For simplicity, if we need more, we add random points in a shell
    if n > 0:
        # Estimate radius for remaining points
        # Mackay icosahedra: 1, 12, 42, 92...
        # If n > 13, we are in shell 2 or higher.
        # Let's just add random points in a shell of radius ~ 2.0 for the rest
        # to ensure we get a diverse structure that relaxes well.
        remaining = n
        for _ in range(remaining):
            r = rng.uniform(1.5, 2.5)
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * r
            points.append(p)
            
    return np.array(points)

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses Simulated Annealing with diverse initializations.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 45.0 # Leave buffer for 6 instances in 300s
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.95 # Cooling rate
    n_iter = 100 # Number of SA steps
    
    # We will run SA from a few different initializations
    inits = []
    
    # 1. Icosahedral
    inits.append(_generate_icosahedral(size, rng))
    
    # 2. Random Shell
    inits.append(_generate_random_shell(size, rng))
    
    # 3. Another Random Shell with different seed
    rng2 = np.random.default_rng(123)
    inits.append(_generate_random_shell(size, rng2))
    
    for init_pos in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb initial position slightly
        pos = init_pos + rng.normal(0, 0.1, init_pos.shape)
        
        # Initial Relaxation
        try:
            energy, pos = relax(pos, potential, parameter)
        except Exception:
            continue
        
        current_energy = energy
        current_pos = pos.copy()
        best_local_energy = energy
        best_local_pos = pos.copy()
        
        T = T_start
        
        for i in range(n_iter):
            if time.time() - start_time > time_limit:
                break
                
            # Kick
            kick_scale = 0.2 + 0.3 * (T / T_start)
            kick = rng.normal(0, kick_scale, pos.shape)
            new_pos = pos + kick
            
            # Relax
            try:
                new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
            except Exception:
                continue
            
            # Acceptance Criterion (Metropolis)
            delta = new_energy - current_energy
            if delta < 0:
                # Accept if better
                current_energy = new_energy
                current_pos = new_pos_relaxed
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_pos = current_pos.copy()
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_energy = new_energy
                        current_pos = new_pos_relaxed
                        if current_energy < best_local_energy:
                            best_local_energy = current_energy
                            best_local_pos = current_pos.copy()
            
            # Cool down
            T *= alpha
            
        # Update global best
        if best_local_energy < best_energy:
            best_energy = best_local_energy
            best_positions = best_local_pos

    if best_positions is None:
        # Absolute fallback
        rng = np.random.default_rng(size)
        radius = 2.0
        points = rng.uniform(-radius, radius, (size, 3))
        try:
            e, best_positions = relax(points, potential, parameter)
        except Exception:
            best_positions = points

    return best_positions

# EVOLVE-BLOCK-END
