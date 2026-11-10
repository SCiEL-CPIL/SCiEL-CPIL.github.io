import numpy as np
import time
import random

N = 256

def _build_aps():
    """Build all 4-term arithmetic progressions as index tuples (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(aps, colors):
    """Count monochromatic 4-APs using numpy for speed."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _count_mono_fast(aps_np, colors):
    """Fast count using numpy array operations."""
    # aps_np shape: (num_aps, 4)
    # colors shape: (256,)
    vals = colors[aps_np]  # shape (num_aps, 4)
    # Check if all 4 are the same: all equal to first
    mono = np.all(vals == vals[:, 0:1], axis=1)
    return int(np.sum(mono))

def _build_aps_np():
    """Build APs as numpy array."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(aps, dtype=np.int32)

def _evaluate(colors, aps_np):
    """Evaluate coloring - return number of mono 4-APs."""
    return _count_mono_fast(aps_np, colors)

def _local_search(colors, aps_np, max_iter=5000, deadline=None):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = colors.copy()
    n = len(colors)
    best_score = _evaluate(colors, aps_np)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping each bit, find the best improvement
        best_delta = 0
        best_idx = -1
        for i in range(n):
            colors[i] = 1 - colors[i]
            new_score = _evaluate(colors, aps_np)
            delta = best_score - new_score
            if delta > best_delta:
                best_delta = delta
                best_idx = i
            colors[i] = 1 - colors[i]  # revert
        
        if best_idx >= 0:
            colors[best_idx] = 1 - colors[best_idx]
            best_score -= best_delta
            improved = True
        else:
            break
    
    return colors, best_score

def _simulated_annealing(colors, aps_np, start_temp=10.0, end_temp=0.1, 
                          max_iter=20000, deadline=None):
    """Simulated annealing to escape local minima."""
    colors = colors.copy()
    current_score = _evaluate(colors, aps_np)
    best_colors = colors.copy()
    best_score = current_score
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Random flip
        idx = random.randint(0, N - 1)
        colors[idx] = 1 - colors[idx]
        new_score = _evaluate(colors, aps_np)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            colors[idx] = 1 - colors[idx]  # revert
    
    return best_colors, best_score

def construct():
    """Construct a 2-coloring of [1,256] minimizing monochromatic 4-term APs."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Build APs once
    aps_np = _build_aps_np()
    
    # Multiple random restarts with local search and simulated annealing
    best_colors = None
    best_score = float('inf')
    
    # Initial diverse starting points
    random.seed(42)
    
    num_restarts = 0
    while time.time() < deadline:
        # Create a random initial coloring
        colors = np.random.randint(0, 2, N).astype(np.int8)
        
        # Phase 1: Local search
        colors, score = _local_search(colors, aps_np, max_iter=1000, deadline=deadline)
        
        # Phase 2: Simulated annealing
        colors, score = _simulated_annealing(colors, aps_np, start_temp=5.0, 
                                               end_temp=0.05, max_iter=5000, 
                                               deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_colors = colors.copy()
        
        num_restarts += 1
        
        if time.time() > deadline - 5:
            break
    
    # Final local search on best solution
    if best_colors is not None and time.time() < deadline - 2:
        best_colors, best_score = _local_search(best_colors, aps_np, max_iter=2000, deadline=deadline)
    
    # Return as list of ints
    return [int(c) for c in best_colors]
