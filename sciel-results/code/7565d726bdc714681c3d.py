import numpy as np
import time
import random

def design(problem):
    size = problem["pixels"]
    half = size // 2
    
    def expand(fd):
        cell = np.zeros((size, size), dtype=int)
        for i in range(half):
            for j in range(half):
                val = fd[i, j]
                cell[i, j] = val
                cell[size - 1 - i, j] = val
                cell[i, size - 1 - j] = val
                cell[size - 1 - i, size - 1 - j] = val
        return cell

    # Precompute mask for proxy score
    # We want to maximize |coeffs|^2 for low-order vectors
    # Let's include vectors with m^2+n^2 <= 9 (up to 3rd order) to be safe
    mask = np.zeros((size, size), dtype=bool)
    for m in range(size):
        for n in range(size):
            freq_m = m if m < size // 2 else m - size
            freq_n = n if n < size // 2 else n - size
            if 1 <= freq_m**2 + freq_n**2 <= 9:
                mask[m, n] = True

    def proxy_score(cell):
        coeffs = np.fft.fft2(cell)
        return np.sum(np.abs(coeffs[mask])**2)

    start_time = time.time()
    deadline = start_time + 110
    
    # Define a few initial patterns
    def init_checkerboard():
        fd = np.zeros((half, half), dtype=int)
        for i in range(half):
            for j in range(half):
                fd[i, j] = 1 if ((i // 2) + (j // 2)) % 2 == 0 else 0
        return fd

    def init_cross():
        fd = np.zeros((half, half), dtype=int)
        # Central cross in the full cell corresponds to central lines in FD
        # Full cell center is between 15 and 16.
        # In FD (0..15), the "center" lines are 15? No.
        # Full cell i=15,16 are the central lines.
        # FD i=15 maps to full cell i=15 and i=16.
        # So setting fd[:, 15] and fd[15, :] to 1 creates a cross?
        # Let's just set a block in the corner? No, cross is central.
        # Let's try a different init: Random
        return None

    def init_random():
        return np.random.randint(0, 2, (half, half))

    inits = [init_checkerboard(), init_random()]
    # Add a "ring" pattern?
    def init_ring():
        fd = np.zeros((half, half), dtype=int)
        for i in range(half):
            for j in range(half):
                # Distance from corner (0,0) in FD?
                # Full cell distance from center.
                # Let's skip complex inits, random is good for SA.
                pass
        return fd

    best_overall_score = -1
    best_overall_cell = None
    
    # Run SA for each init, splitting time
    time_per_init = (deadline - start_time) / len(inits)
    
    for fd_init in inits:
        if fd_init is None: continue
        current_fd = fd_init.copy()
        current_cell = expand(current_fd)
        current_score = proxy_score(current_cell)
        
        local_best_cell = current_cell.copy()
        local_best_score = current_score
        
        T = 1.0
        cooling_rate = 0.999
        init_time = time.time()
        
        while time.time() < init_time + time_per_init:
            i = random.randint(0, half - 1)
            j = random.randint(0, half - 1)
            current_fd[i, j] = 1 - current_fd[i, j]
            
            new_cell = expand(current_fd)
            new_score = proxy_score(new_cell)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_cell = new_cell
                current_score = new_score
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_cell = new_cell.copy()
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        if local_best_score > best_overall_score:
            best_overall_score = local_best_score
            best_overall_cell = local_best_cell.copy()
            
    if best_overall_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_overall_cell[i, j]))
        result.append(row)
    return result
