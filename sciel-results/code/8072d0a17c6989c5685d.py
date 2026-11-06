"""Two-colourings of [1,256] with few monochromatic 4-term APs."""
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a, d >= 1, a+3d <= n
    # Using 0-indexed: positions 0..n-1 correspond to integers 1..n
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # For each position, store list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Count monochromatic 4-APs given a coloring (array of 0/1)
    def count_mono(color):
        count = 0
        for idx in range(num_aps):
            p0, p1, p2, p3 = aps[idx]
            c = color[p0]
            if c == color[p1] and c == color[p2] and c == color[p3]:
                count += 1
        return count
    
    # Thue-Morse initialization (known to be good for avoiding monochromatic APs)
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    # Initialize with Thue-Morse
    best_color = np.array(thue_morse(n), dtype=np.int8)
    best_score = count_mono(best_color)
    
    # Local search: flip one position at a time if it improves
    def local_search(color, score, max_iters=5000):
        color = color.copy()
        score = score
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try flipping each position
            for pos in range(n):
                if time.time() > deadline:
                    return color, score
                old_c = color[pos]
                new_c = 1 - old_c
                
                # Compute delta in monochromatic count
                delta = 0
                for (ap_idx, offset) in pos_aps[pos]:
                    p0, p1, p2, p3 = aps[ap_idx]
                    # Check current state of this AP
                    mono_before = (color[p0] == color[p1] == color[p2] == color[p3])
                    # After flipping pos
                    # The color at pos changes
                    if mono_before:
                        # Was mono, now not mono (since we changed one element)
                        delta -= 1
                    else:
                        # Was not mono. Check if it becomes mono after flip
                        # The other three positions must all have the new color
                        other_positions = [p for p in (p0, p1, p2, p3) if p != pos]
                        if all(color[p] == new_c for p in other_positions):
                            delta += 1
                
                if delta < 0:
                    color[pos] = new_c
                    score += delta
                    improved = True
                    break  # restart scan after improvement
                # If delta == 0, we could also flip (neutral), but let's only take improvements
                # Actually, let's also allow neutral flips with some probability to escape plateaus
                elif delta == 0 and random.random() < 0.1:
                    color[pos] = new_c
                    # score unchanged
                    improved = True
                    break
        
        return color, score
    
    # Run local search from Thue-Morse
    best_color, best_score = local_search(best_color, best_score, max_iters=10000)
    
    # Multi-start: try several random initializations
    random.seed(42)
    np.random.seed(42)
    
    for start_iter in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with bias toward Thue-Morse
        if start_iter < 5:
            # Perturbed Thue-Morse
            color = best_color.copy()
            for _ in range(random.randint(5, 20)):
                color[random.randint(0, n - 1)] = 1 - color[random.randint(0, n - 1)]
        elif start_iter < 15:
            # Random
            color = np.random.randint(0, 2, n, dtype=np.int8)
        else:
            # Thue-Morse with random perturbations
            color = np.array(thue_morse(n), dtype=np.int8)
            for _ in range(random.randint(10, 40)):
                color[random.randint(0, n - 1)] = 1 - color[random.randint(0, n - 1)]
        
        score = count_mono(color)
        
        # Local search
        color, score = local_search(color, score, max_iters=5000)
        
        if score < best_score:
            best_score = score
            best_color = color.copy()
    
    # Final polish: more iterations of local search on the best
    best_color, best_score = local_search(best_color, best_score, max_iters=20000)
    
    # Return as list of ints
    return [int(c) for c in best_color]
