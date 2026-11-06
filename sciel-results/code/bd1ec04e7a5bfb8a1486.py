"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 symmetric photonic-crystal unit cell maximizing the
    gap between TM bands 4 and 5 using a checkerboard pattern with
    strong (2,0) and (0,2) Fourier components.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s budget
    
    # Pattern: cos(2*pi*2*i/size) * cos(2*pi*2*j/size) > threshold
    # This creates a 4x4 checkerboard of (size/4)x(size/4) blocks
    # Maximizing the (2,0) and (0,2) Fourier components for TM band gap
    
    cos_x = [math.cos(2 * math.pi * 2 * i / size) for i in range(size)]
    cos_y = [math.cos(2 * math.pi * 2 * j / size) for j in range(size)]
    
    # Try multiple thresholds to optimize filling fraction
    # For TM band gaps, optimal filling is typically 0.3-0.5
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_cell = None
    best_score = -1.0
    eps_delta = eps_high - eps_low  # 11
    
    for thresh in thresholds:
        if time.time() > deadline:
            break
        
        cell = []
        fourier_20 = 0.0
        for i in range(size):
            row = []
            for j in range(size):
                val = cos_x[i] * cos_y[j]
                pixel = 1 if val > thresh else 0
                row.append(pixel)
                # Accumulate (2,0) Fourier component of epsilon
                eps_val = eps_low + pixel * eps_delta
                fourier_20 += eps_val * cos_x[i]
            
            cell.append(row)
        
        # The gap magnitude is proportional to |Fourier(2,0)|
        score = abs(fourier_20)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Fallback: default checkerboard with threshold 0
    if best_cell is None:
        best_cell = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(1 if cos_x[i] * cos_y[j] > 0 else 0)
            best_cell.append(row)
    
    return best_cell
# EVOLVE-BLOCK-END
