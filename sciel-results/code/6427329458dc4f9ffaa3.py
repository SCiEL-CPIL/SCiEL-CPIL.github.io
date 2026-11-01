import numpy as np
import time

try:
    import science_chemistry
except ImportError:
    science_chemistry = None

N = 55

def construct():
    best_energy = np.inf
    best_positions = None
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    rng = np.random.default_rng(42)
    
    # We will run as many iterations as possible within the time limit.
    # Each iteration: Init/Perturb -> Relax -> Check -> Score
    
    while time.time() - start_time < max_time:
        # 1. Determine Initial Positions for this iteration
        if best_positions is None:
            # First iteration: Random points in a sphere of radius 4.0
            # This is a "liquid" start
            radii = rng.uniform(0, 4.0, N)
            u = rng.random(N)
            v = rng.random(N)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = radii * np.sin(phi) * np.cos(theta)
            y = radii * np.sin(phi) * np.sin(theta)
            z = radii * np.cos(phi)
            
            positions = np.column_stack((x, y, z))
        else:
            # Subsequent iterations: Perturb the best found so far
            # Use a temperature that cools down slightly or stays constant
            # A constant small noise is often effective for basin hopping
            noise_scale = 0.2
            positions = best_positions + rng.normal(0, noise_scale, (N, 3))
            
            # Quick check for overlaps to avoid singularities in relax
            # If overlaps exist, add more noise
            for _ in range(5):
                min_d = np.inf
                # Vectorized min distance check is complex for all pairs, loop is fine for N=55
                for a in range(N):
                    for b in range(a+1, N):
                        d = np.linalg.norm(positions[a] - positions[b])
                        if d < min_d:
                            min_d = d
                if min_d > 0.5:
                    break
                positions += rng.normal(0, 0.1, (N, 3))

        # 2. Relax
        if science_chemistry is None:
            break
            
        try:
            relaxed_pos = science_chemistry.relax(positions)
        except Exception:
            continue

        # 3. Validate Constraint: min distance >= 0.5
        valid = True
        for a in range(N):
            for b in range(a+1, N):
                if np.linalg.norm(relaxed_pos[a] - relaxed_pos[b]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue

        # 4. Compute Energy
        energy = compute_lj_energy(relaxed_pos)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos.copy()
            
            # Optional: If we found a very good energy (close to -279), we might stop early?
            # But we don't know the exact global min value to compare against for early stopping.
            # We can stop if energy < -279.0 (very good) to save time?
            # The known min is -279.248. If we get below -279.2, it's likely the global min.
            if best_energy < -279.2:
                break

    if best_positions is None:
        # Fallback: Return a valid random structure if no relax succeeded
        # Generate a simple grid-like structure that satisfies dist > 0.5
        # 4x4x4 grid = 64 points, take 55.
        # Spacing 1.0 is > 0.5.
        coords = []
        count = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if count < N:
                        coords.append([i, j, k])
                        count += 1
        best_positions = np.array(coords)
        
    return best_positions

def compute_lj_energy(pos):
    """
    Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6)
    """
    n = pos.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r2 = np.sum((pos[i] - pos[j])**2)
            if r2 < 1e-12:
                return np.inf
            r6 = r2**-3
            r12 = r6*r6
            energy += 4.0 * (r12 - r6)
    return energy
