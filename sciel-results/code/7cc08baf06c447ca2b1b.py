"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3(heights):
    """Compute the C3 upper bound: max |autoconvolution| / (sum |heights|)^2."""
    # Autoconvolution via FFT
    n = len(heights)
    f = np.fft.fft(heights, n=2*n)
    autoconv = np.fft.ifft(f * np.conj(f)).real[:n]
    peak = np.max(np.abs(autoconv))
    norm = np.sum(np.abs(heights))
    if norm == 0:
        return float('inf')
    return peak / (norm * norm)


def _flip_bit(heights, idx):
    """Create a copy of heights with the sign flipped at idx."""
    new_heights = heights.copy()
    new_heights[idx] = -new_heights[idx]
    return new_heights


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial candidate: all ones
    best_heights = np.ones(N)
    best_c3 = _compute_c3(best_heights)
    
    # Structured initializations for diversity
    initializations = [
        np.ones(N),
        np.sign(np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))),
        np.sign(np.cos(np.linspace(0, 2*np.pi, N, endpoint=False))),
        np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float),
        np.array([1 if i < N//2 else -1 for i in range(N)], dtype=float),
        np.random.choice([-1, 1], size=N),
    ]
    
    # Evaluate all initializations
    for init in initializations:
        c3 = _compute_c3(init)
        if c3 < best_c3:
            best_c3 = c3
            best_heights = init.copy()
    
    # Discrete hill-climbing refinement
    heights = best_heights.copy()
    c3 = best_c3
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for idx in range(N):
            if time.time() >= deadline:
                break
            new_heights = _flip_bit(heights, idx)
            new_c3 = _compute_c3(new_heights)
            if new_c3 < c3:
                heights = new_heights
                c3 = new_c3
                best_heights = heights.copy()
                best_c3 = c3
                improved = True
                no_improve_count = 0
                break  # Restart scan after an improvement
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    # Final polish: multiple passes with random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try a few random perturbations
        for restart in range(3):
            if time.time() >= deadline:
                break
            perturbed = heights.copy()
            # Flip a small random subset
            n_flips = np.random.randint(1, 10)
            flip_indices = np.random.choice(N, size=n_flips, replace=False)
            perturbed[flip_indices] *= -1
            pert_c3 = _compute_c3(perturbed)
            if pert_c3 < best_c3:
                heights = perturbed
                c3 = pert_c3
                # Refine from here
                no_improve_count = 0
                while time.time() < deadline:
                    improved = False
                    for idx in range(N):
                        if time.time() >= deadline:
                            break
                        new_heights = _flip_bit(heights, idx)
                        new_c3 = _compute_c3(new_heights)
                        if new_c3 < c3:
                            heights = new_heights
                            c3 = new_c3
                            best_heights = heights.copy()
                            best_c3 = c3
                            improved = True
                            no_improve_count = 0
                            break
                    if not improved:
                        no_improve_count += 1
                        if no_improve_count >= max_no_improve:
                            break
    
    return best_heights
# EVOLVE-BLOCK-END
