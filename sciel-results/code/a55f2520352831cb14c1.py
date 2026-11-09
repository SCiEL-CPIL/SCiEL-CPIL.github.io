"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    progressions = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j1 = i + d
            j2 = i + 2 * d
            j3 = i + 3 * d
            if j3 < n:
                progressions.append((i, j1, j2, j3))
    
    # For each position, store which progressions it's in
    pos_to_progs = defaultdict(list)
    for idx, (a, b, c, d) in enumerate(progressions):
        pos_to_progs[a].append(idx)
        pos_to_progs[b].append(idx)
        pos_to_progs[c].append(idx)
        pos_to_progs[d].append(idx)
    
    def validate_fast(arr):
        """Fast validation using precomputed progressions."""
        for a, b, c, d in progressions:
            if len({arr[a], arr[b], arr[c], arr[d]}) < 3:
                return False
        return True
    
    def validate_partial(arr, pos, new_color):
        """Check if changing arr[pos] to new_color maintains validity.
        Only checks progressions involving pos."""
        arr[pos] = new_color
        for pidx in pos_to_progs[pos]:
            a, b, c, d = progressions[pidx]
            if len({arr[a], arr[b], arr[c], arr[d]}) < 3:
                arr[pos] = arr[pos]  # revert
                return False
        return True
    
    def count_colors(arr):
        return len(set(arr))
    
    # Greedy construction
    def greedy_construct():
        colors = [0] * n
        for x in range(n):
            for color in range(n):
                colors[x] = color
                valid = True
                for pidx in pos_to_progs[x]:
                    a, b, c, d = progressions[pidx]
                    if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                        valid = False
                        break
                if valid:
                    break
            else:
                # Should not happen
                colors[x] = 0
        return colors
    
    best_colors = greedy_construct()
    best_count = count_colors(best_colors)
    
    # Local search: try to eliminate colors one by one
    def try_eliminate_color(arr, target_color, deadline):
        """Try to recolor all positions with target_color to other colors."""
        arr = list(arr)
        positions_with_color = [i for i in range(n) if arr[i] == target_color]
        if not positions_with_color:
            return arr, True
        
        # Try to recolor each position
        success = True
        for pos in positions_with_color:
            found = False
            # Try all other colors currently in use
            other_colors = [c for c in set(arr) if c != target_color]
            random.shuffle(other_colors)
            for new_color in other_colors:
                if validate_partial(arr, pos, new_color):
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            return arr, True
        return list(arr), False
    
    # Main local search loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_colors = list(best_colors)
        color_counts = defaultdict(int)
        for c in current_colors:
            color_counts[c] += 1
        
        # Sort colors by frequency (try to eliminate rarest first)
        sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
        
        for target_color in sorted_colors:
            if time.time() > deadline - 5:
                break
            
            new_arr, success = try_eliminate_color(current_colors, target_color, deadline)
            if success:
                new_count = count_colors(new_arr)
                if new_count < best_count:
                    best_colors = new_arr
                    best_count = new_count
                    improved = True
                    break
    
    # Additional local search: try random recoloring to explore
    if time.time() < deadline - 10:
        random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            trial = list(best_colors)
            # Randomly perturb a few positions
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                pos = random.randint(0, n - 1)
                trial[pos] = random.randint(0, n - 1)
            
            if not validate_fast(trial):
                continue
            
            # Try to reduce colors
            trial_count = count_colors(trial)
            if trial_count >= best_count:
                continue
            
            # Try to eliminate colors
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                color_counts = defaultdict(int)
                for c in trial:
                    color_counts[c] += 1
                sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
                found_elim = False
                for target_color in sorted_colors:
                    new_arr, success = try_eliminate_color(trial, target_color, deadline)
                    if success:
                        new_count = count_colors(new_arr)
                        if new_count < trial_count:
                            trial = new_arr
                            trial_count = new_count
                            found_elim = True
                            break
                if not found_elim:
                    break
            
            if trial_count < best_count:
                best_colors = trial
                best_count = trial_count
    
    return best_colors
# EVOLVE-BLOCK-END
