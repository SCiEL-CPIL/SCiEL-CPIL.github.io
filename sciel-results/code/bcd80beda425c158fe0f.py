import time
import numpy as np
import random

# EVOLVE-BLOCK-START

BASE = 51975

# Precompute all proper divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
# Excluding 51975 itself based on seed performance analysis
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1 and v < BASE:
                    _DIVS.append(v)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute positions for each divisor and residue class
# _POS[i][r] = np.array of positions in [0, BASE) where pos % m == r
_POS = []
for m in _DIVS:
    arr = []
    for r in range(m):
        arr.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(arr)

# Map modulus to index
_MOD_TO_IDX = {m: i for i, m in enumerate(_DIVS)}

# Sort indices by coverage (small m first covers more residues)
_COV = [BASE // m for m in _DIVS]
_IDX_BY_COV = sorted(range(N_DIVS), key=lambda i: _COV[i], reverse=True)


def _update_counts(counts, m, r, sign):
    """Update counts array for adding (+1) or removing (-1) a class (m, r)."""
    idx = _MOD_TO_IDX[m]
    pos = _POS[idx][r]
    counts[pos] += sign


def _get_gain(counts, m, r):
    """Calculate gain (number of newly covered residues) for adding (m, r)."""
    idx = _MOD_TO_IDX[m]
    pos = _POS[idx][r]
    return int((counts[pos] == 0).sum())


def _get_loss(counts, m, r):
    """Calculate loss (number of residues becoming uncovered) for removing (m, r)."""
    idx = _MOD_TO_IDX[m]
    pos = _POS[idx][r]
    return int((counts[pos] == 1).sum())


def _get_delta(counts, m, r_old, r_new):
    """Calculate change in uncovered count when changing (m, r_old) to (m, r_new).
    Negative delta means improvement (fewer uncovered)."""
    idx = _MOD_TO_IDX[m]
    pos_old = _POS[idx][r_old]
    pos_new = _POS[idx][r_new]
    
    # Positions in pos_new that are currently uncovered (count 0) will become covered.
    gain = int((counts[pos_new] == 0).sum())
    
    # Positions in pos_old that are currently covered ONLY by this class (count 1) will become uncovered.
    # Note: positions in intersection are handled correctly?
    # If p is in both:
    #   Old state: count(p) includes this class.
    #   New state: count(p) includes this class.
    #   Net change in count is 0.
    #   If count(p) was 1, it was 1 before, becomes 1 after. No change in uncovered status.
    #   If count(p) was >1, it was >1 before, becomes >1 after. No change.
    # So we can just sum gain from new and loss from old.
    loss = int((counts[pos_old] == 1).sum())
    
    return loss - gain


def _greedy_construct(deadline):
    """Greedy construction using coverage counts."""
    counts = np.zeros(BASE, dtype=np.int16)
    classes = []
    used_mods = set()
    
    while time.time() < deadline:
        best_gain = 0
        best_m = -1
        best_r = -1
        
        # Iterate through unused moduli
        for m in _DIVS:
            if m in used_mods:
                continue
            
            idx = _MOD_TO_IDX[m]
            # Find best residue for this modulus
            max_gain_for_m = -1
            best_r_for_m = -1
            
            # Optimization: If m is large, sampling residues might be needed, 
            # but m < 17325, so iterating all residues is feasible if done sparingly.
            # However, doing this for ALL moduli at EVERY step is O(N_divs * M_max).
            # 47 * 17325 ~ 800k operations per step. 
            # Number of steps <= 47. Total ~ 37M ops. Feasible in Python? Maybe slow.
            # Let's optimize: only check moduli that might be useful.
            
            # For speed, we can just check the best residue for each m.
            for r in range(m):
                gain = _get_gain(counts, m, r)
                if gain > max_gain_for_m:
                    max_gain_for_m = gain
                    best_r_for_m = r
            
            if max_gain_for_m > best_gain:
                best_gain = max_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_mods.add(best_m)
        _update_counts(counts, best_m, best_r, 1)
        
    uncovered = int((counts == 0).sum())
    return classes, counts, uncovered


def _local_search(classes, counts, deadline, max_iterations=1000):
    """Local search with Add, Change, and Remove moves."""
    best_classes = [c[:] for c in classes]
    best_counts = counts.copy()
    best_uncovered = int((best_counts == 0).sum())
    used_mods = set(m for m, r in best_classes)
    
    no_improvement_count = 0
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1. Try Change moves
        # For each class, try to change its residue
        for i in range(len(best_classes)):
            if time.time() > deadline:
                break
            m, r = best_classes[i]
            idx = _MOD_TO_IDX[m]
            
            best_delta = 0 # We want delta < 0 for improvement
            best_new_r = r
            
            # Check all other residues
            for r_new in range(m):
                if r_new == r:
                    continue
                delta = _get_delta(best_counts, m, r, r_new)
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = r_new
            
            if best_delta < 0:
                # Apply change
                _update_counts(best_counts, m, r, -1)
                _update_counts(best_counts, m, best_new_r, 1)
                best_classes[i] = [m, best_new_r]
                best_uncovered += best_delta
                improved = True
        
        # 2. Try Add moves
        # For each unused modulus, find best residue
        best_add_gain = 0
        best_add_m = -1
        best_add_r = -1
        
        for m in _DIVS:
            if m in used_mods:
                continue
            
            idx = _MOD_TO_IDX[m]
            max_gain_for_m = -1
            best_r_for_m = -1
            
            for r in range(m):
                gain = _get_gain(best_counts, m, r)
                if gain > max_gain_for_m:
                    max_gain_for_m = gain
                    best_r_for_m = r
            
            if max_gain_for_m > best_add_gain:
                best_add_gain = max_gain_for_m
                best_add_m = m
                best_add_r = best_r_for_m
        
        if best_add_gain > 0:
            # Apply add
            best_classes.append([best_add_m, best_add_r])
            used_mods.add(best_add_m)
            _update_counts(best_counts, best_add_m, best_add_r, 1)
            best_uncovered -= best_add_gain
            improved = True
        
        # 3. Try Remove moves (only if it doesn't increase uncovered, i.e., gain 0)
        # Usually not useful for minimization, but might free up space? 
        # No, moduli are distinct, so removing doesn't allow adding another.
        # It just reduces class count. Skip.

        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 5:
                break
        else:
            no_improvement_count = 0
            
    return best_classes, best_counts, best_uncovered


def construct():
    deadline = time.time() + 115.0
    best_classes = []
    best_uncovered = BASE
    best_counts = np.zeros(BASE, dtype=np.int16)
    
    # Strategy 1: Greedy + Local Search
    classes, counts, unc = _greedy_construct(deadline - 20.0)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes[:]
        best_counts = counts.copy()
    
    if time.time() < deadline - 10.0 and best_classes:
        classes, counts, unc = _local_search(best_classes, best_counts, deadline - 5.0)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes[:]
            best_counts = counts.copy()
            
    # Strategy 2: Randomized Greedy + Local Search
    if time.time() < deadline - 30.0:
        random.seed(42)
        for attempt in range(10):
            if time.time() > deadline - 15.0:
                break
            
            # Shuffle moduli order for greedy
            shuffled_divs = _DIVS[:]
            random.shuffle(shuffled_divs)
            
            # Modify _DIVS temporarily? No, _greedy_construct uses global _DIVS.
            # We need a variant of greedy that accepts an order.
            
            # Let's implement a quick randomized greedy inline
            counts_r = np.zeros(BASE, dtype=np.int16)
            classes_r = []
            used_mods_r = set()
            
            # Randomly select a subset of moduli to start with?
            # Or just shuffle the order of consideration.
            
            # Re-implement greedy with shuffled order
            # For efficiency, we'll just pick random unused moduli in each step
            for _ in range(len(_DIVS)):
                if time.time() > deadline - 15.0:
                    break
                available = [m for m in _DIVS if m not in used_mods_r]
                if not available:
                    break
                m = random.choice(available)
                
                idx = _MOD_TO_IDX[m]
                best_r = 0
                best_gain = -1
                for r in range(m):
                    gain = _get_gain(counts_r, m, r)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_gain > 0:
                    classes_r.append([m, best_r])
                    used_mods_r.add(m)
                    _update_counts(counts_r, m, best_r, 1)
                else:
                    # If no gain, maybe skip this modulus for this attempt
                    # But we might still want to try others
                    pass
            
            unc_r = int((counts_r == 0).sum())
            
            # Local search on this randomized start
            if time.time() < deadline - 10.0 and classes_r:
                classes_r, counts_r, unc_r = _local_search(classes_r, counts_r, deadline - 5.0)
            
            if unc_r < best_uncovered:
                best_uncovered = unc_r
                best_classes = classes_r[:]
                best_counts = counts_r.copy()
                
    return best_classes

# EVOLVE-BLOCK-END
