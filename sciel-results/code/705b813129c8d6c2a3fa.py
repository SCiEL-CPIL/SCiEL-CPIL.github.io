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

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    divs = _divisors_of(BASE)
    all_moduli = divs
    
    # Greedy Initialization
    # Sort descending to cover large chunks first
    mods_sorted = sorted(all_moduli, reverse=True)
    
    ref_count = np.zeros(BASE, dtype=np.int32)
    classes = []
    
    for m in mods_sorted:
        best_r = 0
        best_gain = -1
        # For each residue, check how many uncovered it covers
        # Optimization: if m is large, N/m is small. If m is small, N/m is large.
        # But we must check all r in [0, m).
        # Cost: sum(m) over all moduli? No, cost is sum over m of (m * (N/m)) = K * N.
        # This is acceptable for initialization.
        for r in range(m):
            # Count uncovered in this progression
            gain = np.sum(ref_count[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        ref_count[best_r::m] += 1
        
    current_uncovered = int(np.sum(ref_count == 0))
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute indices for each modulus? 
    # No, slicing is faster.
    
    n_classes = len(classes)
    indices = list(range(n_classes))
    
    # We can speed up by pre-computing the arithmetic progressions?
    # np.arange(r, BASE, m) is fast.
    
    while time.time() < deadline and T > T_min:
        # Pick a random modulus
        idx = random.choice(indices)
        m, r_old = classes[idx]
        
        # Pick a random new residue
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            continue
            
        # Calculate delta
        # Note: ref_count includes the current class (m, r_old)
        # So for r_new (disjoint from r_old), ref_count[r_new::m] does NOT include this class.
        # For r_old, ref_count[r_old::m] DOES include this class.
        
        pos_old = np.arange(r_old, BASE, m)
        pos_new = np.arange(r_new, BASE, m)
        
        # Loss: indices in old set that have ref_count == 1 (only covered by this class)
        loss = np.sum(ref_count[pos_old] == 1)
        
        # Gain: indices in new set that have ref_count == 0 (uncovered)
        gain = np.sum(ref_count[pos_new] == 0)
        
        delta = loss - gain
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            ref_count[pos_old] -= 1
            ref_count[pos_new] += 1
            current_uncovered += delta
            classes[idx][1] = r_new
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
        T *= cooling_rate
        
    return best_classes
