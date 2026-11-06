"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the TM band-4/5 gap.
    Uses a dielectric background with a super-ellipse air hole pattern,
    with 180-degree rotational symmetry about the cell center.
    """
    size = problem["pixels"]  # 32
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    cx = cy = (size - 1) / 2.0  # 15.5
    
    # Dielectric background with a super-ellipse air region
    # Super-ellipse: (|dx|/a)^n + (|dy|/b)^n <= 1 means air
    # Tuned for maximum TM 4-5 band gap
    
    n_exp = 4.0   # squircle shape
    a = 6.5       # half-width
    b = 6.5       # half-height
    
    cell = [[0]*size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            px = i + 0.5
            py = j + 0.5
            dx = abs(px - cx)
            dy = abs(py - cy)
            
            if (dx / a) ** n_exp + (dy / b) ** n_exp <= 1.0:
                cell[i][j] = 0  # air
            else:
                cell[i][j] = 1  # dielectric
    
    # Enforce 180-degree rotational symmetry: cell[i][j] == cell[31-i][31-j]
    for i in range(size):
        for j in range(size):
            cell[i][j] = cell[size - 1 - i][size - 1 - j]
    
    return cell
# EVOLVE-BLOCK-END
