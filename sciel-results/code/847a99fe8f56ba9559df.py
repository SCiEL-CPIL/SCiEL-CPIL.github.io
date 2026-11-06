import time
import random
import numpy as np

N = 128
DEADLINE = 58.0  # Safety margin before 60s limit

def _get_progressions():
    """Generate all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 1..N."""
    ps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ps.append((a - 1, b - 1, c - 1, e - 1)) # 0-based indices
    return ps

PROGS = _get_progressions()

# Precompute for each position, which progressions it is part of
POS_TO_PROGS = [[] for _ in range(N)]
for i, p in enumerate(PROGS):
    for pos in p:
        POS_TO_PROGS[pos].append(i)

def _count_colors_in_prog(p, colors):
    """Count distinct colors in a progression."""
    return len({colors[x] for x in p})

def _is_bad(p, colors):
    """True if progression uses fewer than 3 distinct colors."""
    return _count_colors_in_prog(p, colors) < 3

def construct():
    t_start = time.time()
    
    # We search for a valid coloring with the minimum number of colors.
    # The theoretical minimum is 3.
    # If 3 colors is impossible, we try 4, 5, etc.
    
    max_palette = 10
    best_found = None
    
    for palette_size in range(3, max_palette + 1):
        if time.time() - t_start > DEADLINE - 5:
            break
            
        # Try multiple restarts for each palette size
        restarts = 5 if palette_size == 3 else 2
        
        for r in range(restarts):
            if time.time() - t_start > DEADLINE - 2:
                break
                
            current_colors = np.random.randint(0, palette_size, size=N)
            
            # Calculate initial violations
            violations = 0
            for i, p in enumerate(PROGS):
                if _is_bad(p, current_colors):
                    violations += 1
            
            if violations == 0:
                return current_colors.tolist()
                
            # Local Search: Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.9995
            max_steps = 50000
            step = 0
            
            while violations > 0 and time.time() - t_start < DEADLINE - 1:
                step += 1
                if step % 1000 == 0:
                    T *= alpha
                    if T < T_min:
                        break
                        
                # Pick a random position to change
                pos = np.random.randint(0, N)
                old_color = current_colors[pos]
                
                # Pick a new color
                new_color = np.random.randint(0, palette_size)
                if new_color == old_color:
                    new_color = (new_color + 1) % palette_size
                    
                # Calculate delta in violations
                delta_viol = 0
                for pi in POS_TO_PROGS[pos]:
                    p = PROGS[pi]
                    old_count = _count_colors_in_prog(p, current_colors)
                    old_bad = 1 if old_count < 3 else 0
                    
                    current_colors[pos] = new_color
                    new_count = _count_colors_in_prog(p, current_colors)
                    new_bad = 1 if new_count < 3 else 0
                    
                    delta_viol += (new_bad - old_bad)
                    
                # Acceptance criterion
                if delta_viol < 0:
                    violations += delta_viol
                elif delta_viol == 0:
                    if random.random() < 0.5:
                        pass # Keep
                    else:
                        current_colors[pos] = old_color
                else:
                    if random.random() < np.exp(-delta_viol / T):
                        violations += delta_viol
                    else:
                        current_colors[pos] = old_color
                
                if violations == 0:
                    break
                    
            if violations == 0:
                return current_colors.tolist()
                
    # If we fail to find a valid coloring, return the last state (will score 0 if invalid)
    return current_colors.tolist()
