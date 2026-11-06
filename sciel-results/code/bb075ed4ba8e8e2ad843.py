"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit
    
    # Symmetry group: D4 (square symmetry)
    # Generate orbits of pixels under D4 symmetry
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = set()
            # Generate all symmetric equivalents
            candidates = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (j, size - 1 - i),
                (size - 1 - i, j),
                (size - 1 - j, i)
            ]
            unique_candidates = []
            for c in candidates:
                if c not in visited and c not in unique_candidates:
                    unique_candidates.append(c)
            
            for c in unique_candidates:
                orbit.add(c)
                visited.add(c)
            if orbit:
                orbits.append(list(orbit))
    
    # Sort orbits by size for determinism
    orbits.sort(key=len)
    num_orbits = len(orbits)
    
    # Fast Surrogate: Use FFT to measure strength of fundamental periodicity
    def compute_surrogate_gap(cell_np):
        fill_frac = np.sum(cell_np) / (size * size)
        if fill_frac < 0.3 or fill_frac > 0.7:
            return -1.0
        
        # Compute FFT
        fft_cell = np.fft.fft2(cell_np)
        
        # Magnitude of fundamental components
        c10 = np.abs(fft_cell[1, 0])
        c01 = np.abs(fft_cell[0, 1])
        c11 = np.abs(fft_cell[1, 1])
        
        # We want strong fundamental modulation
        score = c10 + c01 + c11
        return score

    # Local Search with Random Restarts
    best_score = -np.inf
    best_cell = None
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random symmetric pattern
        orbit_vals = np.random.randint(0, 2, num_orbits)
        cell_np = np.zeros((size, size), dtype=int)
        for k, orbit in enumerate(orbits):
            for (i, j) in orbit:
                cell_np[i, j] = orbit_vals[k]
        
        current_score = compute_surrogate_gap(cell_np)
        
        # Hill Climbing
        for step in range(2000):
            if time.time() > deadline - 5:
                break
                
            # Pick a random orbit to flip
            idx = np.random.randint(num_orbits)
            
            # Flip
            old_val = orbit_vals[idx]
            orbit_vals[idx] = 1 - old_val
            
            # Update cell
            for (i, j) in orbits[idx]:
                cell_np[i, j] = orbit_vals[idx]
            
            # Evaluate
            new_score = compute_surrogate_gap(cell_np)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = cell_np.copy()
            else:
                # Revert
                orbit_vals[idx] = old_val
                for (i, j) in orbits[idx]:
                    cell_np[i, j] = old_val
    
    # If no best cell found, return a default symmetric cell
    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                if abs(i + 0.5 - center) < 8 and abs(j + 0.5 - center) < 8:
                    best_cell[i, j] = 1
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
