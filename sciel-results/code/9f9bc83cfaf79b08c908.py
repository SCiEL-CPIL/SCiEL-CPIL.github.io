"""
Optimized Morse Cluster Minimizer using Simulated Annealing with Langevin Dynamics.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 295.0
start_time_global = time.time()

def sa_minimize(n, rho, time_budget):
    """
    Perform Simulated Annealing with Langevin Dynamics to minimize Morse potential energy.
    """
    rng = np.random.default_rng()
    
    # Initial parameters
    # Radius of initial sphere
    R = 0.5 * n ** (1.0/3.0) + 2.0
    T_start = 2.0
    T_end = 0.05
    dt = 0.02
    gamma = 0.8  # Damping factor
    mass = 1.0
    
    # Track best solution
    best_energy = np.inf
    best_pos = None
    
    # State variables
    # Initialize positions randomly in a sphere
    def random_sphere(radius):
        # Box-Muller or rejection
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check min distance
                if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
        return np.array(points)
    
    pos = random_sphere(R)
    vel = np.zeros((n, 3))
    
    # Calculate initial energy
    try:
        E, grad = morse_energy_gradient(pos, rho)
        # Check constraints
        # We assume morse_energy_gradient returns (E, gradient) where gradient is dE/dx
        # Force = -gradient
    except Exception:
        # Fallback if gradient fails
        return np.array([np.array([i, 0, 0]) for i in range(n)], dtype=float)
        
    if not np.isfinite(E):
        return np.array([np.array([i, 0, 0]) for i in range(n)], dtype=float)
        
    best_energy = E
    best_pos = pos.copy()
    
    start_time = time.time()
    total_steps = 0
    no_improve_steps = 0
    max_no_improve = 5000 # Restart if stuck
    
    # We need to estimate total steps for cooling schedule
    # Let's run for a fixed number of steps per "phase" or just use time
    # Simpler: run until time budget. Cool T exponentially based on steps taken?
    # Better: Cool T based on time elapsed.
    
    while time.time() - start_time < time_budget:
        # Global time check
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 2.0:
            break
            
        # Determine current temperature
        # Exponential cooling: T = T_start * (T_end/T_start)^(elapsed / time_budget)
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_budget)
        T = T_start * (T_end / T_start) ** progress
        
        # Langevin Step
        # F = -grad
        F = -grad
        # Noise
        noise = rng.normal(0, 1, (n, 3))
        
        # Update velocity
        # v_new = gamma * v + F * dt / m + sqrt(2 * T * dt / m) * noise
        # m = 1
        noise_scale = np.sqrt(2.0 * T * dt)
        vel = gamma * vel + F * dt + noise_scale * noise
        
        # Update position
        new_pos = pos + vel * dt
        
        # Check minimum distance constraint
        # Efficient check: only if n is small or using KDTree? 
        # For n=80, O(n^2) is 6400 checks, which is fast.
        valid = True
        # Quick check: compute all pair distances
        # dists = np.linalg.norm(new_pos[:, np.newaxis] - new_pos[np.newaxis, :], axis=2)
        # if np.min(dists[np.triu_indices(n, k=1)]) < 0.5: valid = False
        
        # Optimization: Check only if we suspect violation? 
        # Actually, Morse potential is very steep. Violation leads to huge energy.
        # Let's compute energy first, and if it's huge, reject.
        # But constraint is hard: "atoms must be at least 0.5 apart".
        # So we must enforce it.
        
        # Compute distances for constraint
        # To speed up, we can skip full distance check if we know we are far apart? 
        # No, just do it.
        try:
            # Compute pairwise distances
            dists = np.linalg.norm(new_pos[:, np.newaxis] - new_pos[np.newaxis, :], axis=2)
            min_dist = np.min(dists[np.triu_indices(n, k=1)])
            if min_dist < 0.5:
                valid = False
        except Exception:
            valid = False
            
        if not valid:
            # Reject move, keep old pos/vel? Or just pos?
            # In Langevin, if we reject, we usually keep old state.
            # But we already updated vel. Let's revert vel to old? 
            # Standard Metropolis for Langevin: propose new (x, v). If rejected, keep old (x, v).
            # So we don't update pos or vel.
            continue
            
        # Calculate new energy
        try:
            new_E, new_grad = morse_energy_gradient(new_pos, rho)
        except Exception:
            continue
            
        if not np.isfinite(new_E):
            continue
            
        # Metropolis Acceptance
        dE = new_E - E
        if dE < 0:
            accept = True
        else:
            # Avoid overflow
            if dE > 100 * T:
                accept = False
            else:
                accept = np.log(rng.uniform()) > -dE / T
                
        if accept:
            pos = new_pos
            vel = vel # vel is already updated in proposal
            E = new_E
            grad = new_grad
            total_steps += 1
            
            if E < best_energy:
                best_energy = E
                best_pos = pos.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Reject: pos and vel remain unchanged
            # We don't increment total_steps for cooling? 
            # Usually cooling is based on time or accepted steps. 
            # Let's base it on time.
            no_improve_steps += 1
            
        # Restart logic
        if no_improve_steps > max_no_improve:
            # Restart with new random position and higher T
            pos = random_sphere(R)
            vel = np.zeros((n, 3))
            try:
                E, grad = morse_energy_gradient(pos, rho)
            except Exception:
                break
            if not np.isfinite(E):
                break
            # Reset T to a mid-value
            T = T_start * 0.5
            no_improve_steps = 0
            # Note: The cooling schedule in the while loop depends on elapsed time, 
            # so T will continue to cool down. This might be too fast.
            # Ideally, we reset the clock or adjust T. 
            # For simplicity, we just restart and let the time-based cooling continue.
            
    # Final relaxation of the best found position to ensure it's a local minimum
    if best_pos is not None:
        try:
            final_E, final_pos = relax(best_pos, 'morse', rho)
            if np.isfinite(final_E):
                return final_pos
        except Exception:
            pass
            
    return best_pos

def construct(instance):
    global start_time_global
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Time check
    elapsed = time.time() - start_time_global
    if elapsed > TOTAL_TIME_BUDGET - 5.0:
        # Emergency
        positions = np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float)
        try:
            _, positions = relax(positions, potential, rho)
        except:
            pass
        return positions

    # Determine time budget for this instance
    # Assume equal distribution of remaining time for remaining instances?
    # We don't know how many are left. 
    # Heuristic: Use 1/6 of total budget if we are early, less if late.
    # Since we don't know the index, let's use a safe fraction of remaining time.
    # If this is the first call, remaining is ~295. Budget ~49s.
    # If this is the last call, remaining is small.
    # Let's assume we get ~45s per instance on average.
    # If we have used a lot of time, we have less.
    remaining_total = TOTAL_TIME_BUDGET - elapsed
    inst_budget = min(45.0, remaining_total * 0.4)
    
    # Ensure minimum budget
    if inst_budget < 5.0:
        inst_budget = 2.0 # Very tight, just do a quick random search
        
    return sa_minimize(size, rho, inst_budget)

# EVOLVE-BLOCK-END
