"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N, d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a-1, a+d-1, a+2*d-1, a+3*d-1))  # 0-indexed
    
    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # Build reverse index: for each position, which APs contain it
    pos_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    # Function to calculate score (number of monochromatic 4-APs)
    def get_score(colours):
        # colours is a numpy array of 0s and 1s
        count = 0
        for i in range(num_aps):
            a, b, c, d = aps[i]
            if colours[a] == colours[b] == colours[c] == colours[d]:
                count += 1
        return count
    
    # Faster score using vectorization
    def get_score_fast(colours):
        # colours is a numpy array of 0s and 1s
        # For each AP, check if all four positions have the same colour
        # colours[aps] has shape (num_aps, 4)
        cols = colours[aps]
        # Check if all elements in each row are equal
        # A row is monochromatic if min == max
        min_col = cols.min(axis=1)
        max_col = cols.max(axis=1)
        mono = (min_col == max_col).sum()
        return int(mono)
    
    # Delta computation: when flipping position p, only APs containing p are affected
    def compute_delta(colours, p):
        """Compute change in score if we flip position p."""
        delta = 0
        for ap_idx in pos_to_aps[p]:
            a, b, c, d = aps[ap_idx]
            # Count how many of the four positions currently have the same colour as p
            same_count = 0
            for pos in (a, b, c, d):
                if colours[pos] == colours[p]:
                    same_count += 1
            
            # Before flip: AP is monochromatic if same_count == 4
            # After flip: the four colours become (4-same_count) of new colour, same_count of old
            # After flip, AP is monochromatic if 4-same_count == 4, i.e., same_count == 0
            before_mono = 1 if same_count == 4 else 0
            after_mono = 1 if same_count == 0 else 0
            delta += after_mono - before_mono
        return delta
    
    # Greedy local search: repeatedly flip the position that gives the best improvement
    def greedy_local_search(colours, max_steps=10000):
        score = get_score_fast(colours)
        for step in range(max_steps):
            if time.time() > deadline:
                break
            best_delta = 0
            best_pos = -1
            # Try all positions
            for p in range(N):
                d = compute_delta(colours, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            if best_pos == -1 or best_delta >= 0:
                break
            colours[best_pos] ^= 1
            score += best_delta
        return colours, score
    
    # Simulated annealing with delta computation
    def simulated_annealing(colours, t_start=1.0, t_end=0.01, n_steps=50000):
        score = get_score_fast(colours)
        current = colours.copy()
        best = colours.copy()
        best_score = score
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Pick a random position to flip
            p = random.randint(0, N - 1)
            delta = compute_delta(current, p)
            
            # Acceptance criterion
            if delta <= 0 or random.random() < np.exp(-delta / t_start * (1 - step / n_steps)):
                current[p] ^= 1
                score += delta
                if score < best_score:
                    best_score = score
                    best = current.copy()
            
            # Cool down
            # t = t_start * (t_end / t_start) ** (step / n_steps)
        
        return best, best_score
    
    # Run multiple strategies
    best_overall = None
    best_overall_score = float('inf')
    
    # Strategy 1: Multiple random restarts with greedy local search
    n_restarts = 0
    while time.time() < deadline - 5.0:
        # Random initial coloring
        colours = np.random.randint(0, 2, N).astype(np.int8)
        
        # Greedy local search
        colours, score = greedy_local_search(colours, max_steps=20000)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall = colours.copy()
        
        n_restarts += 1
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Strategy 2: Simulated annealing from the best greedy result
    if best_overall is not None and time.time() < deadline - 5.0:
        sa_result, sa_score = simulated_annealing(best_overall, t_start=0.5, t_end=0.01, n_steps=30000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_result
    
    # Final: try a few more greedy refinements on the best result
    if best_overall is not None and time.time() < deadline - 2.0:
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            refined, refined_score = greedy_local_search(best_overall.copy(), max_steps=5000)
            if refined_score < best_overall_score:
                best_overall_score = refined_score
                best_overall = refined
    
    # Return as list of ints
    if best_overall is None:
        return [0] * N
    return best_overall.tolist()
# EVOLVE-BLOCK-END
