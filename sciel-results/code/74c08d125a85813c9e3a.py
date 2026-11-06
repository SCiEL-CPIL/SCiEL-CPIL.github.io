"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def design(problem):
    """
    Design a 32x32 unit cell with four tunable superellipse dielectric regions
    to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    - Use a geometric parameterization (4 superellipses at ±0.25, ±0.25)
      which naturally has C4v symmetry and strong 2nd-order Fourier components.
    - Optimize shape parameters (exponent, size, offset) using a fast
      FFT-based surrogate objective that targets the relevant Fourier modes.
    - Enforce exact symmetry by construction and verification.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s budget
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # Precompute coordinate grid
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='xy')
    
    def make_cell_from_params(params):
        """
        params: [n_exp, a_size, offset]
        - n_exp: superellipse exponent (1=square, 2=circle, >2=rounded square)
        - a_size: half-width of superellipse
        - offset: distance of centers from origin (±offset, ±offset)
        
        Returns a 32x32 nested list with exact C4v symmetry.
        """
        n_exp = params[0]
        a_size = params[1]
        offset = params[2]
        
        cell = np.zeros((size, size), dtype=int)
        
        # Four centers at (±offset, ±offset)
        centers = [
            (offset, offset),
            (-offset, offset),
            (offset, -offset),
            (-offset, -offset)
        ]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            # Superellipse condition: |dx/a|^n + |dy/a|^n <= 1
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
        
        # Enforce exact C4v symmetry by averaging over all 8 symmetry operations
        sym_cell = np.zeros((size, size), dtype=float)
        sym_cell += cell
        sym_cell += cell.T  # transpose (reflection across y=x)
        sym_cell += cell[::-1, :]  # vertical flip
        sym_cell += cell[:, ::-1]  # horizontal flip
        sym_cell += np.rot90(cell, 1)  # 90° rotation
        sym_cell += np.rot90(cell, 2)  # 180° rotation
        sym_cell += np.rot90(cell, 3)  # 270° rotation
        # The 8th symmetry is the anti-diagonal reflection, which is:
        # flip both axes then transpose, or equivalently rot90(transpose)
        sym_cell += np.rot90(cell.T, 1)
        
        # Threshold at 4 (half of 8) to get binary result
        sym_cell = (sym_cell >= 4.0).astype(int)
        
        return sym_cell.tolist()
    
    def fitness(cell_list):
        """
        Surrogate: maximize the 2nd-order Fourier components of 1/epsilon,
        which drive the 4-5 TM band gap.
        """
        cell_arr = np.array(cell_list)
        eps_arr = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_arr
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Key Fourier components for 4-5 TM gap:
        # (2,0), (0,2) and their negatives - these couple bands 4 and 5
        m_20 = np.abs(fft_inv_eps[2, 0])
        m_02 = np.abs(fft_inv_eps[0, 2])
        m_m20 = np.abs(fft_inv_eps[size-2, 0])
        m_0m2 = np.abs(fft_inv_eps[0, size-2])
        
        # (2,2) also contributes
        m_22 = np.abs(fft_inv_eps[2, 2])
        m_m2m2 = np.abs(fft_inv_eps[size-2, size-2])
        
        # (1,1) for lower band structure
        m_11 = np.abs(fft_inv_eps[1, 1])
        
        # Weighted score emphasizing 2nd-order modes
        score = 4.0 * (m_20 + m_02 + m_m20 + m_0m2) + 2.0 * (m_22 + m_m2m2) + 0.5 * m_11
        return score
    
    # Parameter ranges
    # n_exp: [1.0, 4.0] - square to rounded square
    # a_size: [0.05, 0.20] - small to large superellipse
    # offset: [0.10, 0.35] - distance from center
    
    best_params = None
    best_score = -1.0
    
    # Multi-start local search
    num_starts = 30
    steps_per_start = 3000
    
    for s in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize parameters
        if s == 0:
            # Start with a reasonable guess: circular rods (n=2), moderate size
            params = [2.0, 0.12, 0.25]
        elif s == 1:
            # Slightly rounded square
            params = [1.5, 0.14, 0.25]
        elif s == 2:
            # More rounded
            params = [3.0, 0.10, 0.25]
        else:
            params = [
                random.uniform(1.0, 4.0),
                random.uniform(0.05, 0.20),
                random.uniform(0.10, 0.35)
            ]
        
        current_score = fitness(make_cell_from_params(params))
        if current_score > best_score:
            best_score = current_score
            best_params = params[:]
        
        # Local search: perturb random parameter
        base_step = 0.05
        for step in range(steps_per_start):
            if time.time() > deadline - 5.0:
                break
            
            # Decay step size
            step_size = base_step * (0.95 ** (step // 100))
            
            # Pick a random parameter to perturb
            pidx = random.randint(0, 2)
            old_val = params[pidx]
            
            # Perturb with Gaussian noise
            if pidx == 0:  # n_exp
                params[pidx] = max(1.0, min(4.0, old_val + random.gauss(0, step_size)))
            elif pidx == 1:  # a_size
                params[pidx] = max(0.05, min(0.20, old_val + random.gauss(0, step_size * 0.5)))
            else:  # offset
                params[pidx] = max(0.10, min(0.35, old_val + random.gauss(0, step_size * 0.3)))
            
            new_score = fitness(make_cell_from_params(params))
            
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = params[:]
            else:
                params[pidx] = old_val
    
    if best_params is None:
        # Fallback: simple four-rod design
        return make_cell_from_params([2.0, 0.12, 0.25])
    
    return make_cell_from_params(best_params)

# EVOLVE-BLOCK-END
