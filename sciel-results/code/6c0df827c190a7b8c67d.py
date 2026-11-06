"""Simulated annealing for minimizing distinct colors in 4-AP 3-coloring constraint."""
import time
import random

# EVOLVE-BLOCK-START
def _count_violations(colors, n=128):
    """Count 4-term progressions with fewer than 3 distinct colors."""
    count = 0
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            prog = {colors[a], colors[b], colors[c], colors[e]}
            if len(prog) < 3:
                count += 1
    return count

def _count_distinct(colors):
    return len(set(colors))

def _simulated_annealing(seed, deadline, max_colors=8):
    """Simulated annealing to minimize distinct colors while satisfying constraints."""
    rng = random.Random(seed)
    n = 128
    
    # Initialize with random assignment using max_colors colors
    colors = [rng.randint(0, max_colors - 1) for _ in range(n)]
    
    # Precompute all 4-term progressions
    progressions = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                progressions.append((a, b, c, e))
    
    # Count how many progressions each position participates in
    pos_count = [0] * n
    for (a, b, c, e) in progressions:
        pos_count[a] += 1
        pos_count[b] += 1
        pos_count[c] += 1
        pos_count[e] += 1
    
    # Initial state
    current_colors = list(colors)
    current_violations = _count_violations(current_colors, n)
    current_distinct = _count_distinct(current_colors)
    
    best_colors = list(current_colors)
    best_score = current_violations * 1000 + current_distinct
    
    # Temperature schedule
    T0 = 50.0
    T_end = 0.01
    total_steps = 50000
    
    start_time = time.time()
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
        
        # Temperature
        T = T0 * (T_end / T0) ** (step / total_steps)
        
        # Pick a random position, weighted toward high-constraint positions
        # but also allowing low-constraint positions for exploration
        if rng.random() < 0.3:
            # Weighted selection
            pos = rng.choices(range(n), weights=pos_count, k=1)[0]
        else:
            pos = rng.randint(0, n - 1)
        
        old_color = current_colors[pos]
        
        # Try changing to a different color
        # Prefer colors already in use to reduce distinct count
        used_colors = set(current_colors)
        if used_colors:
            candidate_colors = [c for c in used_colors if c != old_color]
            if not candidate_colors:
                continue
            new_color = rng.choice(candidate_colors)
        else:
            new_color = rng.randint(0, max_colors - 1)
        
        if new_color == old_color:
            continue
        
        # Compute delta in violations by checking progressions involving pos
        delta_violations = 0
        for (a, b, c, e) in progressions:
            if pos not in (a, b, c, e):
                continue
            # Old violation status
            old_prog = {current_colors[a], current_colors[b], current_colors[c], current_colors[e]}
            old_bad = len(old_prog) < 3
            
            # New violation status
            new_colors_temp = list(current_colors)
            new_colors_temp[pos] = new_color
            new_prog = {new_colors_temp[a], new_colors_temp[b], new_colors_temp[c], new_colors_temp[e]}
            new_bad = len(new_prog) < 3
            
            if old_bad and not new_bad:
                delta_violations -= 1
            elif not old_bad and new_bad:
                delta_violations += 1
        
        # Compute delta in distinct colors
        old_distinct = current_distinct
        # Remove old_color from consideration, add new_color
        color_counts = {}
        for c in current_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        color_counts[old_color] -= 1
        color_counts[new_color] = color_counts.get(new_color, 0) + 1
        new_distinct = sum(1 for v in color_counts.values() if v > 0)
        delta_distinct = new_distinct - old_distinct
        
        # Score delta (violations weighted heavily)
        delta_score = delta_violations * 1000 + delta_distinct
        
        # Accept or reject
        if delta_score <= 0 or rng.random() < (2.718281828 ** (-delta_score / max(T, 1e-10))):
            current_colors[pos] = new_color
            current_violations += delta_violations
            current_distinct = new_distinct
            
            current_score = current_violations * 1000 + current_distinct
            if current_score < best_score:
                best_score = current_score
                best_colors = list(current_colors)
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 128
    best_solution = None
    best_count = 128
    
    # Run multiple simulated annealing runs with different seeds and max_colors
    for max_colors in [4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        n_restarts = min(50, int(remaining / 2))
        
        for seed in range(n_restarts):
            if time.time() > deadline:
                break
            
            colors = _simulated_annealing(seed, deadline, max_colors)
            if colors is not None:
                violations = _count_violations(colors, n)
                if violations == 0:
                    count = _count_distinct(colors)
                    if count < best_count:
                        best_count = count
                        best_solution = list(colors)
                        # If we found a very good solution, we can stop early
                        if count <= 3:
                            break
        
        if best_count <= 3:
            break
    
    # If no valid solution found, try a simple greedy fallback
    if best_solution is None:
        colors = [0] * n
        for x in range(n):
            found = False
            for color in range(8):
                colors[x] = color
                valid = True
                for d in range(1, x // 3 + 1):
                    a = x - 3 * d
                    if a < 0:
                        break
                    prog = {colors[a], colors[a+d], colors[a+2*d], colors[x]}
                    if len(prog) < 3:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                colors[x] = 0
        best_solution = colors
        best_count = _count_distinct(best_solution)
    
    return best_solution
# EVOLVE-BLOCK-END
