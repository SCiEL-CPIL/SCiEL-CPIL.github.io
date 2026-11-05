"""Search approach: Basin hopping with Langevin dynamics and annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _langevin_step(positions, velocities, rho, dt, temp, damping):
    """
    Perform a single Langevin dynamics step.
    Uses velocity Verlet-like integration with stochastic force.
    """
    n = len(positions)
    # Compute gradient (force is -gradient)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
    except Exception:
        return positions, velocities, energy
    
    # Force = -gradient
    forces = -grad
    
    # Update velocities: v(t+dt) = v(t) * exp(-gamma*dt) + sqrt(2*kT*gamma/m) * dt * N(0,1)
    # Mass m = 1, k_B = 1
    # Noise amplitude: sqrt(2 * temp * damping / dt) * dt = sqrt(2 * temp * damping * dt)
    noise_scale = math.sqrt(2.0 * temp * damping * dt) if temp > 0 else 0.0
    
    # Update positions
    new_positions = positions + velocities * dt
    
    # Update velocities
    # v_new = v * exp(-damping*dt) + forces*dt + noise
    # For small dt, exp(-damping*dt) ~ 1 - damping*dt
    decay = math.exp(-damping * dt)
    new_velocities = velocities * decay + forces * dt + noise_scale * np.random.randn(n, 3)
    
    return new_positions, new_velocities, energy

def construct(instance):
    """
    Basin hopping search:
    1. Start with a compact random configuration.
    2. Perform Langevin dynamics with high temperature to explore.
    3. Periodically cool down and perform local relaxation.
    4. Keep the best result found.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_energy = np.inf
    best_positions = None
    
    # Initial configuration: Random points in a sphere of radius ~ sqrt(n)
    # Ensure minimum distance by rejection sampling
    radius = math.sqrt(size) * 1.0
    initial_pos = []
    attempts = 0
    max_attempts = size * 100
    while len(initial_pos) < size and attempts < max_attempts:
        p = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in initial_pos:
                if np.linalg.norm(p - q) < 0.6:
                    valid = False
                    break
            if valid:
                initial_pos.append(p)
        attempts += 1
    
    # If we didn't get enough, just add random points (relaxation will fix)
    while len(initial_pos) < size:
        p = np.random.uniform(-radius, radius, 3)
        initial_pos.append(p)
    
    current_pos = np.array(initial_pos, dtype=float)
    velocities = np.zeros((size, 3))
    
    # Parameters for Langevin
    dt = 0.01
    damping = 1.0
    
    # Annealing schedule
    # Start hot, cool down over time
    # We will run in cycles: Explore (Langevin) -> Relax (Local Minimization)
    # Each cycle reduces temperature.
    
    # Estimate time per Langevin step
    # n=80, pairs ~ 3160. Gradient computation is O(n^2).
    # Let's assume ~1ms per step. 
    # Total budget ~ 290s.
    # We can do ~100k steps? Maybe too many.
    # Let's do batches of 500 Langevin steps, then 1 relaxation.
    
    temp = 2.0  # Initial temperature
    cooling_rate = 0.95  # Multiply temp by this each cycle
    
    # To ensure we get a valid answer quickly, do one relaxation immediately
    try:
        e, rel_pos = relax(current_pos, potential, rho)
        if np.isfinite(e):
            best_energy = e
            best_positions = rel_pos.copy()
            current_pos = rel_pos.copy()
            velocities = np.zeros((size, 3))
    except Exception:
        pass
    
    while time.time() < deadline:
        # Check if we should stop
        if time.time() > deadline - 10:
            break
            
        # Determine batch size based on remaining time
        # Assume 1ms per Langevin step (conservative for n=80)
        # Relaxation takes ~10-50ms depending on convergence.
        # Let's try 100 Langevin steps per cycle.
        n_steps = 100
        
        # Run Langevin dynamics
        for _ in range(n_steps):
            if time.time() > deadline - 5:
                break
            current_pos, velocities, e = _langevin_step(current_pos, velocities, rho, dt, temp, damping)
            # Simple constraint: keep atoms within a reasonable bound to prevent explosion
            # If any atom goes too far, reset velocity and pull back? 
            # Or just let it be, relaxation will fix.
            # However, if energy becomes NaN or Inf, break.
            if not np.isfinite(e):
                # Reset to last good position if we have one, or current
                break
        
        # After Langevin steps, attempt local relaxation
        try:
            e, rel_pos = relax(current_pos, potential, rho)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = rel_pos.copy()
                current_pos = rel_pos.copy()
                velocities = np.zeros((size, 3))
            else:
                # If relaxation failed or didn't improve, keep current_pos for next round
                # But update velocities to zero to avoid momentum carrying over weirdly
                velocities = np.zeros((size, 3))
        except Exception:
            pass
        
        # Cool down
        temp *= cooling_rate
        
        # If temperature is very low, do a larger random kick (basin hopping move)
        if temp < 0.01:
            # Random perturbation
            kick_size = 0.5
            current_pos += np.random.uniform(-kick_size, kick_size, size=(size, 3))
            velocities = np.zeros((size, 3))
            # Reheat slightly to allow escape
            temp = 0.5 * cooling_rate ** 0.5  # Restart annealing from a moderate temp
            
    # Final relaxation to ensure best possible
    if best_positions is not None:
        try:
            e, rel_pos = relax(best_positions, potential, rho)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = rel_pos.copy()
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_positions is None:
        # Linear chain
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i, 0, 0]
        try:
            best_energy, best_positions = relax(best_positions, potential, rho)
        except Exception:
            pass

    # Final validity check: ensure min distance
    # The relax function usually enforces this, but let's be safe.
    # If there are violations, the score will be 0 or negative? 
    # "atoms must be at least 0.5 apart". 
    # If relax returns a state with violations, it's invalid.
    # Let's verify.
    valid = True
    for i in range(size):
        for j in range(i + 1, size):
            if np.linalg.norm(best_positions[i] - best_positions[j]) < 0.5:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to fix by pushing apart
        for _ in range(50):
            violations = False
            for i in range(size):
                for j in range(i + 1, size):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < 0.5:
                        violations = True
                        diff = best_positions[i] - best_positions[j]
                        norm = np.linalg.norm(diff)
                        if norm > 1e-10:
                            push = diff / norm * (0.5 - d) * 0.5
                            best_positions[i] += push
                            best_positions[j] -= push
            if not violations:
                break
        # Re-relax after fixing?
        try:
            e, rel_pos = relax(best_positions, potential, rho)
            if np.isfinite(e):
                # Check validity again
                v2 = True
                for i in range(size):
                    for j in range(i + 1, size):
                        if np.linalg.norm(rel_pos[i] - rel_pos[j]) < 0.5:
                            v2 = False
                            break
                    if not v2:
                        break
                if v2:
                    best_positions = rel_pos
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
