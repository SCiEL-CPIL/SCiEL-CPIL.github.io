import numpy as np

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell using optimized Perlin noise thresholding.
    """
    size = problem["pixels"]
    
    # Parameters tuned for TM band gap 4-5
    # Higher frequency features often help with higher order gaps
    base_freq = 4.0 
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    seed = 123 # Changed seed to explore different texture
    
    # 1. Generate raw noise field
    # We use a simple value noise with smooth interpolation.
    # To speed up, we pre-calculate the grid.
    
    def get_noise_value(x, y, s):
        # x, y are float coordinates in noise space
        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - ix, y - iy
        
        # Smoothstep
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        
        def hash_2d(ix, iy):
            h = (ix * 374761393 + iy * 668265263 + s * 1442695041) & 0x7FFFFFFF
            return h / 0x7FFFFFFF
        
        v00 = hash_2d(ix, iy)
        v10 = hash_2d(ix + 1, iy)
        v01 = hash_2d(ix, iy + 1)
        v11 = hash_2d(ix + 1, iy + 1)
        
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        return v0 + sy * (v1 - v0)

    raw_noise = np.zeros((size, size), dtype=np.float64)
    
    # Generate multi-scale noise
    for octave in range(octaves):
        freq = base_freq * (lacunarity ** octave)
        amp = persistence ** octave
        
        for i in range(size):
            for j in range(size):
                # Map pixel coords to noise space
                # We want the noise to be C4v symmetric *after* symmetrization, 
                # so we can sample non-symmetric noise.
                # However, sampling symmetric points in the noise space 
                # ensures the raw noise has some symmetry, which helps.
                # Let's just sample raw noise at (i/size * freq, j/size * freq)
                # and symmetrize later.
                
                nx = (i / size) * freq
                ny = (j / size) * freq
                
                raw_noise[i, j] += amp * get_noise_value(nx, ny, seed + octave)

    # 2. Enforce C4v Symmetry
    # The group C4v has 8 elements.
    # We average the field and its 7 symmetric counterparts.
    
    # Define the 8 transformations on the indices (i, j)
    # Center is (15.5, 15.5) for size 32.
    # Transformation: (i, j) -> (i', j')
    
    # 1. Identity: (i, j)
    # 2. Rot 90: (j, size-1-i)  ? Let's check.
    #    If center is C, and vector is v, Rot 90 is v_perp.
    #    Pixel (i, j). Center (C, C).
    #    u = i - C, v = j - C.
    #    Rot 90 CCW: u' = -v, v' = u.
    #    i' = C - v = C - (j - C) = 2C - j.
    #    j' = C + u = C + (i - C) = i.
    #    So (i, j) -> (2C - j, i).
    #    For size 32, C = 15.5.
    #    i' = 31 - j, j' = i.
    
    # 3. Rot 180: (2C - i, 2C - j) -> (31 - i, 31 - j)
    # 4. Rot 270: (j, 2C - i) -> (j, 31 - i)
    
    # 5. Refl X (Vertical axis): (i, 2C - j) -> (i, 31 - j)
    # 6. Refl Y (Horizontal axis): (2C - i, j) -> (31 - i, j)
    # 7. Refl Diag (y=x): (j, i)
    # 8. Refl Anti-Diag (y=-x): (2C - j, 2C - i) -> (31 - j, 31 - i)

    # We can use NumPy advanced indexing to average these.
    # Create index arrays
    i_idx, j_idx = np.indices((size, size))
    
    # Calculate symmetric indices
    sym_indices = [
        (i_idx, j_idx),                         # 1
        (31 - j_idx, i_idx),                    # 2 (Rot 90)
        (31 - i_idx, 31 - j_idx),               # 3 (Rot 180)
        (j_idx, 31 - i_idx),                    # 4 (Rot 270)
        (i_idx, 31 - j_idx),                    # 5 (Refl Y)
        (31 - i_idx, j_idx),                    # 6 (Refl X)
        (j_idx, i_idx),                         # 7 (Refl Diag)
        (31 - j_idx, 31 - i_idx)                # 8 (Refl Anti-Diag)
    ]
    
    sym_sum = np.zeros((size, size), dtype=np.float64)
    for i_s, j_s in sym_indices:
        sym_sum += raw_noise[i_s, j_s]
        
    sym_noise = sym_sum / 8.0

    # 3. Threshold
    # Normalize to [0, 1]
    min_val = sym_noise.min()
    max_val = sym_noise.max()
    if max_val > min_val:
        norm_noise = (sym_noise - min_val) / (max_val - min_val)
    else:
        norm_noise = sym_noise
        
    # Threshold
    # Tuning threshold is crucial. 
    # For TM 4-5 gap, fill fraction ~0.5 is common.
    threshold = 0.5
    cell_binary = (norm_noise > threshold).astype(int)
    
    # 4. Final Symmetry Check (just in case of rounding errors, though integer math should be fine)
    # The averaging above ensures float symmetry. Thresholding might break it if values are exactly at threshold.
    # To be safe, enforce symmetry on the binary grid again.
    # This is fast.
    
    # We can use the same index trick or simple flips.
    # Since we averaged 8 points, the values are symmetric.
    # If val > T, then all 8 symmetric vals > T.
    # So the binary grid should be symmetric.
    
    return cell_binary.tolist()
