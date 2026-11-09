"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell using gradient Perlin noise thresholding
    with C4v-symmetric angular modulation to maximize TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0

    # Gradient noise using deterministic hash-based gradients
    def gradient_noise_2d(x, y, seed=0):
        """2D gradient noise with smooth interpolation."""
        def hash_grad(ix, iy):
            h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0x7FFFFFFF
            # Use hash to select gradient direction from 8 directions
            angle = (h % 8) * np.pi / 4.0
            return np.cos(angle), np.sin(angle)

        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - ix, y - iy

        # Quintic smoothstep for smoother interpolation
        sx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        sy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)

        # Get gradient vectors at corners
        gx00, gy00 = hash_grad(ix, iy)
        gx10, gy10 = hash_grad(ix + 1, iy)
        gx01, gy01 = hash_grad(ix, iy + 1)
        gx11, gy11 = hash_grad(ix + 1, iy + 1)

        # Dot products with offset vectors
        v00 = gx00 * fx + gy00 * fy
        v10 = gx10 * (fx - 1) + gy10 * fy
        v01 = gx01 * fx + gy01 * (fy - 1)
        v11 = gx11 * (fx - 1) + gy11 * (fy - 1)

        # Bilinear interpolation
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        return v0 + sy * (v1 - v0)

    # Generate multi-scale C4v-symmetric noise field
    cell = np.zeros((size, size), dtype=np.float64)

    # Multi-scale parameters tuned for TM gap
    scales = [
        (3.0, 1.0, 42),    # (frequency, amplitude, seed)
        (6.0, 0.5, 137),
        (12.0, 0.25, 251),
        (24.0, 0.1, 389),
    ]

    for freq, amp, seed in scales:
        for i in range(size):
            for j in range(size):
                dx = (i - cx) / size
                dy = (j - cy) / size

                # C4v symmetric coordinates: use min/max of |dx|, |dy|
                ax = abs(dx)
                ay = abs(dy)
                sx = min(ax, ay)
                sy = max(ax, ay)

                # Sample noise at the symmetric fundamental domain point
                noise_val = gradient_noise_2d(sx * freq, sy * freq, seed)

                cell[i, j] += amp * noise_val

    # Add C4v-symmetric angular modulation
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / size
            dy = (j - cy) / size
            r = np.sqrt(dx * dx + dy * dy)
            theta = np.arctan2(dy, dx)

            # 4-fold and 8-fold angular terms
            ang4 = np.cos(4 * theta) * np.exp(-r * 8)
            ang8 = np.cos(8 * theta) * np.exp(-r * 12)

            # Radial rings
            radial = np.cos(2 * np.pi * 2.5 * r) * 0.3

            cell[i, j] += 0.15 * ang4 + 0.08 * ang8 + 0.1 * radial

    # Add a gentle radial bias to encourage ring-like structures
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / size
            dy = (j - cy) / size
            r = np.sqrt(dx * dx + dy * dy)
            # Slight preference for mid-radius regions
            cell[i, j] += 0.1 * np.exp(-((r - 0.3) ** 2) / 0.04)

    # Normalize
    cell_min = cell.min()
    cell_max = cell.max()
    if cell_max > cell_min:
        cell_norm = (cell - cell_min) / (cell_max - cell_min)
    else:
        cell_norm = np.zeros_like(cell)

    # Threshold
    threshold = 0.48
    cell_binary = (cell_norm > threshold).astype(int)

    # Enforce C4v symmetry strictly
    # Average over all symmetry operations
    sym_sum = np.zeros_like(cell_binary, dtype=np.float64)
    sym_sum += cell_binary
    sym_sum += cell_binary[:, ::-1]        # vertical flip
    sym_sum += cell_binary[::-1, :]        # horizontal flip
    sym_sum += cell_binary.T               # transpose
    sym_sum += cell_binary[::-1, ::-1]     # 180 rotation
    sym_sum += cell_binary.T[:, ::-1]      # anti-diagonal flip
    sym_sum += cell_binary[::-1, :].T      # another rotation+flip
    sym_sum += cell_binary[::-1, ::-1].T   # another

    cell_binary = (sym_sum / 8.0).round().astype(int)

    # Clean up: remove isolated single pixels (improves manufacturability and band structure)
    from scipy.ndimage import binary_opening, binary_closing
    struct = np.ones((2, 2))
    cell_binary = binary_opening(cell_binary, structure=struct).astype(int)
    cell_binary = binary_closing(cell_binary, structure=struct).astype(int)

    # Re-enforce symmetry after morphological operations
    sym_sum = np.zeros_like(cell_binary, dtype=np.float64)
    sym_sum += cell_binary
    sym_sum += cell_binary[:, ::-1]
    sym_sum += cell_binary[::-1, :]
    sym_sum += cell_binary.T
    sym_sum += cell_binary[::-1, ::-1]
    sym_sum += cell_binary.T[:, ::-1]
    sym_sum += cell_binary[::-1, :].T
    sym_sum += cell_binary[::-1, ::-1].T
    cell_binary = (sym_sum / 8.0).round().astype(int)

    # Ensure center pixel is consistent
    cell_binary[int(cx), int(cy)] = int(cell_binary[int(cx), int(cy)])

    # Convert to nested list
    cell_list = cell_binary.tolist()

    return cell_list
# EVOLVE-BLOCK-END
