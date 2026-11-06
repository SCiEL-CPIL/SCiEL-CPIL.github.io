"""Two-colouring of [1,256] minimizing monochromatic 4-term APs."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0  # safety margin under 120s
    
    # Precompute all 4-term arithmetic progressions
    # AP: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= n, d >= 1
    # Use 0-based indexing: positions 0..n-1
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    def count_mono(color):
        """Count monochromatic 4-term APs given a 0/1 coloring array."""
        # For each AP, check if all 4 positions have the same color
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        # Monochromatic if all equal
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    # Initialize with Thue-Morse sequence (known to have few short APs)
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    best_color = color.copy()
    best_score = count_mono(best_color)
    
    # Local search: try flipping individual bits
    def local_search(color, max_iters=200000):
        color = color.copy()
        score = count_mono(color)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try random flips
            for _ in range(50):
                if time.time() >= deadline:
                    break
                idx = np.random.randint(n)
                new_color = color.copy()
                new_color[idx] = 1 - new_color[idx]
                new_score = count_mono(new_color)
                if new_score < score:
                    color = new_color
                    score = new_score
                    improved = True
                    break
        return color, score
    
    # Multi-start local search
    num_starts = 1
    for start in range(num_starts):
        if time.time() >= deadline - 10:
            break
        
        if start == 0:
            # Start from Thue-Morse
            c = best_color.copy()
        else:
            # Start from random coloring
            c = np.random.randint(0, 2, n).astype(np.int8)
        
        c, s = local_search(c)
        if s < best_score:
            best_color = c
            best_score = s
    
    # Additional refinement: try all single-bit flips
    if time.time() < deadline - 5:
        for idx in range(n):
            if time.time() >= deadline - 2:
                break
            new_color = best_color.copy()
            new_color[idx] = 1 - new_color[idx]
            new_score = count_mono(new_color)
            if new_score < best_score:
                best_color = new_color
                best_score = new_score
    
    # Convert to list of ints
    return [int(x) for x in best_color]
