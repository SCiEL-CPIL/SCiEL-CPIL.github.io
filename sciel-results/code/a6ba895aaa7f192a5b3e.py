"""D4-symmetric cellular-automaton (Game of Life) derived photonic crystal unit cell."""
# EVOLVE-BLOCK-START
import numpy as np


def _enforce_d4(grid):
    """Enforce D4 symmetry about the center by averaging all 8 symmetric copies and thresholding."""
    n = grid.shape[0]
    # All 8 D4 transforms: identity, 90, 180, 270 rot, and their transposes
    transforms = [
        grid,
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        grid.T,
        np.rot90(grid, 1).T,
        np.rot90(grid, 2).T,
        np.rot90(grid, 3).T,
    ]
    avg = np.mean(transforms, axis=0)
    return (avg >= 0.5).astype(int)


def _step_life(grid):
    """One step of Conway's Game of Life with toroidal boundary."""
    n = grid.shape[0]
    # Count neighbors with toroidal wrap
    neighbors = (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )
    # Life rules: live if 2-3 neighbors; die otherwise
    alive = ((neighbors == 2) | (neighbors == 3))
    return alive.astype(int)


def _make_seed_cross(size):
    """Create a small symmetric cross/plus seed at the center."""
    grid = np.zeros((size, size), dtype=int)
    c = size // 2
    # Small cross: 3x3 plus shape
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if abs(dx) + abs(dy) <= 1:
                grid[c + dx, c + dy] = 1
    return grid


def _make_seed_ring(size):
    """Create a small ring seed at the center."""
    grid = np.zeros((size, size), dtype=int)
    c = size // 2
    r = 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - c) ** 2 + (j - c) ** 2)
            if 1.5 < dist < 2.5:
                grid[i, j] = 1
    return grid


def _make_seed_checker_small(size):
    """Create a small checkerboard-like symmetric seed."""
    grid = np.zeros((size, size), dtype=int)
    c = size // 2
    # 5x5 checkerboard in center
    for i in range(c - 2, c + 3):
        for j in range(c - 2, c + 3):
            if (i + j) % 2 == 0:
                grid[i, j] = 1
    return grid


def design(problem):
    """Design a D4-symmetric photonic crystal unit cell using cellular automaton dynamics."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Try multiple seeds and step counts, pick a robust configuration
    best_cell = None
    best_score = -1
    
    seeds = [
        _make_seed_cross(size),
        _make_seed_ring(size),
        _make_seed_checker_small(size),
    ]
    
    for seed in seeds:
        for steps in range(1, 6):
            grid = seed.copy()
            for _ in range(steps):
                grid = _step_life(grid)
                grid = _enforce_d4(grid)
            
            # Check density is reasonable (20-70% dielectric)
            density = grid.mean()
            if density < 0.15 or density > 0.75:
                continue
            
            # Simple heuristic: prefer patterns with good spatial frequency content
            # Compute a rough measure of high-frequency energy
            f = np.fft.fft2(grid)
            fshift = np.fft.fftshift(f)
            # Zero out the DC component
            c = size // 2
            fshift[c, c] = 0
            # Measure energy in outer rings (high frequency)
            yy, xx = np.mgrid[0:size, 0:size]
            r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
            high_freq_mask = r > size * 0.25
            total_energy = np.sum(np.abs(fshift) ** 2)
            high_freq_energy = np.sum(np.abs(fshift[high_freq_mask]) ** 2)
            hf_ratio = high_freq_energy / (total_energy + 1e-10)
            
            # Also check that the pattern isn't too uniform (has variation)
            variation = grid.std()
            
            score = hf_ratio * variation
            if score > best_score:
                best_score = score
                best_cell = grid
    
    if best_cell is None:
        # Fallback: use the cross seed with 2 steps
        grid = _make_seed_cross(size)
        for _ in range(2):
            grid = _step_life(grid)
            grid = _enforce_d4(grid)
        best_cell = grid
    
    # Convert to nested list
    return best_cell.tolist()
# EVOLVE-BLOCK-END
