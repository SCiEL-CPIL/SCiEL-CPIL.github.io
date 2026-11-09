"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

_BASE = 51975

# Precompute all divisors > 1 of 51975
def _get_divisors():
    divs = []
    for d in range(2, _BASE + 1):
        if _BASE % d == 0:
            divs.append(d)
    return divs

_DIVISORS = _get_divisors()
_DIVISORS_SORTED = sorted(_DIVISORS)

def _count_uncovered(covered):
    """Count uncovered residues."""
    return sum(1 for i in range(_BASE) if not covered[i])

def _apply_class(covered, m, r):
    """Mark residues covered by class (m, r)."""
    for i in range(r, _BASE, m):
        covered[i] = 1

def _unapply_class(covered, m, r):
    """Unmark residues covered by class (m, r). Note: This is only valid if we know
    that no other class covers these residues, or if we use a count array.
    For simplicity in the greedy phase, we just recompute or use a count array."""
    pass

def _greedy_best():
    """Greedy construction: at each step, pick (m, r) that covers most uncovered residues."""
    covered = bytearray(_BASE)
    classes = []
    used_moduli = set()
    available_moduli = _DIVISORS_SORTED[:]
    
    start_time = time.time()
    
    while available_moduli and time.time() < start_time + 10:
        best_m = -1
        best_r = -1
        best_gain = -1
        
        # Evaluate all available moduli
        for m in available_moduli:
            # Find best r for this m
            # Optimization: If m is large, the number of residues covered is small.
            # We can estimate max gain as _BASE // m.
            # If max possible gain <= best_gain, skip?
            # Max gain for m is _BASE // m.
            if _BASE // m <= best_gain:
                continue
                
            for r in range(m):
                # Count gain
                gain = 0
                for i in range(r, _BASE, m):
                    if not covered[i]:
                        gain += 1
                        if gain > best_gain:
                            # Early termination? No, we need exact gain to compare, 
                            # but if gain already > best_gain, we can stop counting?
                            # Actually, we just need to know if it's better.
                            # But we need the exact gain to store? No, just best_m, best_r.
                            # However, if we break early, we don't know the full gain, 
                            # but we know it's at least 'gain'. 
                            # If 'gain' > 'best_gain', this is a candidate.
                            # We can continue counting to get the true gain, or just record it.
                            # For correctness, let's just count fully.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_m != -1:
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            available_moduli.remove(best_m)
            for i in range(best_r, _BASE, best_m):
                covered[i] = 1
        else:
            break
            
    return classes, covered

def _local_search(classes, time_limit=100.0):
    """Local search to improve the solution."""
    start_time = time.time()
    
    # Initialize coverage count
    cov_count = [0] * _BASE
    for m, r in classes:
        for i in range(r, _BASE, m):
            cov_count[i] += 1
            
    def get_score():
        return sum(1 for i in range(_BASE) if cov_count[i] == 0)
    
    current_score = get_score()
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    # Map modulus to index in classes list for quick lookup?
    # Or just iterate.
    
    while time.time() < time_limit:
        improved = False
        
        # 1. Try changing residue of existing classes
        for idx, (m, r) in enumerate(classes):
            # Unapply current
            for i in range(r, _BASE, m):
                cov_count[i] -= 1
                
            # Try all other residues
            best_r = r
            best_sc = get_score()
            
            for r_new in range(m):
                if r_new == r:
                    continue
                # Apply candidate
                for i in range(r_new, _BASE, m):
                    cov_count[i] += 1
                sc = get_score()
                if sc < best_sc:
                    best_sc = sc
                    best_r = r_new
                # Unapply candidate
                for i in range(r_new, _BASE, m):
                    cov_count[i] -= 1
            
            # Apply best
            for i in range(best_r, _BASE, m):
                cov_count[i] += 1
                
            if best_r != r:
                classes[idx][1] = best_r
                improved = True
                current_score = best_sc
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    
        # 2. Try adding a new class
        used_moduli = {m for m, r in classes}
        for m_new in _DIVISORS_SORTED:
            if m_new in used_moduli:
                continue
            # Find best r for m_new
            best_r = -1
            best_sc = current_score
            
            for r_new in range(m_new):
                for i in range(r_new, _BASE, m_new):
                    cov_count[i] += 1
                sc = get_score()
                if sc < best_sc:
                    best_sc = sc
                    best_r = r_new
                for i in range(r_new, _BASE, m_new):
                    cov_count[i] -= 1
                    
            if best_r != -1 and best_sc < current_score:
                classes.append([m_new, best_r])
                used_moduli.add(m_new)
                for i in range(best_r, _BASE, m_new):
                    cov_count[i] += 1
                current_score = best_sc
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    
        # 3. Try removing a class
        for idx in range(len(classes) - 1, -1, -1):
            m, r = classes[idx]
            # Unapply
            for i in range(r, _BASE, m):
                cov_count[i] -= 1
            sc = get_score()
            
            if sc <= current_score:
                # If score doesn't increase, remove it (it was redundant or neutral)
                classes.pop(idx)
                current_score = sc
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
            else:
                # Reapply
                for i in range(r, _BASE, m):
                    cov_count[i] += 1
                
        if not improved:
            # No improvement found in this pass. 
            # We could try random restarts or perturbations here.
            # For now, break if no improvement to save time, or continue with randomization.
            # Let's add a simple random perturbation: swap a residue randomly.
            if len(classes) > 0:
                idx = random.randint(0, len(classes) - 1)
                m, r = classes[idx]
                r_new = random.randint(0, m - 1)
                if r_new != r:
                    # Apply change
                    for i in range(r, _BASE, m):
                        cov_count[i] -= 1
                    for i in range(r_new, _BASE, m):
                        cov_count[i] += 1
                    classes[idx][1] = r_new
                    current_score = get_score()
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
            else:
                break

    return best_classes

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Greedy Init
    classes, covered = _greedy_best()
    
    if not classes:
        # Fallback to a simple valid class
        classes = [[3, 0]]
        
    # 2. Local Search
    result = _local_search(classes, time_limit=deadline - 5)
    
    if result and len(result) > 0:
        return result
    else:
        return classes

# EVOLVE-BLOCK-END
