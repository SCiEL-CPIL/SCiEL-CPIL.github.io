import numpy as np
import time
import random
import math

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def _greedy_init(mods_sorted):
    """Fast greedy initialization: pick residue that maximizes uncovered coverage."""
    ref_count = np.zeros(BASE, dtype=np.int32)
    classes = []
    
    for m in mods_sorted:
        best_r = 0
        best_gain = -1
        # Vectorized: compute gains for all residues at once
        # For each r in [0, m), count uncovered in ref_count[r::m]
        # This is m separate slices, but we can do it efficiently
        for r in range(m):
            gain = np.sum(ref_count[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        ref_count[best_r::m] += 1
        
    return classes, ref_count

def _simulated_annealing(classes, ref_count, start_time, deadline, T0=3.0, T_min=0.05, cooling=0.9995):
    """Simulated annealing with fast move evaluation."""
    current_uncovered = int(np.sum(ref_count == 0))
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    n_classes = len(classes)
    indices = list(range(n_classes))
    T = T0
    
    # Precompute arange positions for each class to avoid repeated np.arange
    # But residue changes, so we need dynamic arange. However, we can cache by (m, r)
    # Actually, np.arange is fast enough. Let's focus on the delta computation.
    
    while time.time() < deadline and T > T_min:
        # Pick a random class
        idx = random.choice(indices)
        m, r_old = classes[idx]
        
        # Pick a random new residue
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            continue
        
        # Fast delta computation using numpy slicing
        # pos_old and pos_new are the arithmetic progressions
        # We need: loss = count of ref_count[pos_old] == 1
        #           gain = count of ref_count[pos_new] == 0
        
        # Use slicing directly for speed
        old_slice = ref_count[r_old::m]
        new_slice = ref_count[r_new::m]
        
        # These slices don't include the current class's contribution to new_slice,
        # but do include it in old_slice.
        loss = np.count_nonzero(old_slice == 1)
        gain = np.count_nonzero(new_slice == 0)
        
        delta = loss - gain
        
        # Acceptance: accept if delta < 0 (improvement) or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Apply move
            ref_count[r_old::m] -= 1
            ref_count[r_new::m] += 1
            current_uncovered += delta
            classes[idx][1] = r_new
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
        
        T *= cooling
    
    return best_uncovered, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    mods_sorted = sorted(divs, reverse=True)
    
    best_uncovered_global = float('inf')
    best_classes_global = None
    
    # Multiple restarts to explore different regions
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Greedy initialization
        classes, ref_count = _greedy_init(mods_sorted)
        
        # Simulated annealing with time budget
        restart_deadline = start_time + (restart + 1) * time_per_restart
        uncovered, classes_sa = _simulated_annealing(
            classes, ref_count, start_time, restart_deadline,
            T0=3.0, T_min=0.05, cooling=0.9995
        )
        
        if uncovered < best_uncovered_global:
            best_uncovered_global = uncovered
            best_classes_global = [c[:] for c in classes_sa]
    
    return best_classes_global
