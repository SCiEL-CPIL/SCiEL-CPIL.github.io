import time
import random
import numpy as np

BASE = 51975

def get_moduli():
    moduli = []
    for i in range(3, BASE + 1, 2):
        if BASE % i == 0:
            moduli.append(i)
    return moduli

MODULI = get_moduli()
# Precompute indices for each modulus and residue? No, too much memory.
# Just use np.arange on the fly.

def add_class(counts, uncovered_count, m, r):
    indices = np.arange(r, BASE, m)
    was_uncovered = counts[indices] == 0
    newly_covered = np.sum(was_uncovered)
    counts[indices] += 1
    uncovered_count -= newly_covered
    return uncovered_count

def remove_class(counts, uncovered_count, m, r):
    indices = np.arange(r, BASE, m)
    counts[indices] -= 1
    became_uncovered = counts[indices] == 0
    newly_uncovered = np.sum(became_uncovered)
    uncovered_count += newly_uncovered
    return uncovered_count

def best_residue_for_modulus(counts, m, exclude_r=None):
    """Find residue r for modulus m that maximizes newly covered residues."""
    best_gain = -1
    best_r = 0
    for r in range(m):
        if r == exclude_r:
            continue
        indices = np.arange(r, BASE, m)
        gain = np.sum(counts[indices] == 0)
        if gain > best_gain:
            best_gain = gain
            best_r = r
    return best_r, best_gain

def greedy_init(randomize=True):
    counts = np.zeros(BASE, dtype=np.int32)
    uncovered_count = BASE
    classes = []
    used_moduli = set()
    moduli_list = MODULI.copy()
    if randomize:
        random.shuffle(moduli_list)
    
    for m in moduli_list:
        # Find best residue for this m given current counts
        r, gain = best_residue_for_modulus(counts, m)
        if gain > 0:
            classes.append([m, r])
            used_moduli.add(m)
            uncovered_count = add_class(counts, uncovered_count, m, r)
        # If gain is 0, we might still add it? No, it doesn't help.
        # But maybe it helps in combination with others? 
        # In greedy, we usually only add if it helps immediately.
        # However, a class with 0 immediate gain might be part of a better set?
        # Unlikely to be better than a class with >0 gain.
        
    return classes, counts, uncovered_count

def local_search(classes, counts, uncovered_count, deadline):
    """
    Perform local search to improve the solution.
    Returns improved classes, counts, uncovered_count.
    """
    used_moduli = {c[0] for c in classes}
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Optimize residues for existing classes
        for i in range(len(classes)):
            m, r = classes[i]
            # Remove current
            uncovered_count = remove_class(counts, uncovered_count, m, r)
            
            # Find best new residue
            best_r, best_gain = best_residue_for_modulus(counts, m, exclude_r=r)
            
            # Add best
            if best_gain > 0 or best_r != r:
                # If best_gain is 0, we might still want to change if it allows other swaps?
                # For now, just take the best.
                classes[i] = [m, best_r]
                uncovered_count = add_class(counts, uncovered_count, m, best_r)
                
                if best_r != r:
                    improved = True
            
        # Pass 2: Try adding new classes
        for m in MODULI:
            if m in used_moduli:
                continue
            r, gain = best_residue_for_modulus(counts, m)
            if gain > 0:
                # Add it
                classes.append([m, r])
                used_moduli.add(m)
                uncovered_count = add_class(counts, uncovered_count, m, r)
                improved = True
                break # Restart pass 2 after one addition to update used_moduli properly?
                # Actually, just continue checking others in this pass?
                # No, adding one changes the landscape. Better to restart or just take the best one.
                # Let's take the best one found in the scan.
        
        # Re-do Pass 2 to find the single best addition
        # This is a bit inefficient. Let's just scan and pick the best.
        best_add_m = None
        best_add_r = None
        best_add_gain = 0
        for m in MODULI:
            if m in used_moduli:
                continue
            r, gain = best_residue_for_modulus(counts, m)
            if gain > best_add_gain:
                best_add_gain = gain
                best_add_m = m
                best_add_r = r
        
        if best_add_m is not None:
            classes.append([best_add_m, best_add_r])
            used_moduli.add(best_add_m)
            uncovered_count = add_class(counts, uncovered_count, best_add_m, best_add_r)
            improved = True

        # Pass 3: Try removing classes that contribute 0
        # Check if any class covers 0 unique residues
        for i in range(len(classes)):
            m, r = classes[i]
            indices = np.arange(r, BASE, m)
            unique_covered = np.sum(counts[indices] == 1)
            if unique_covered == 0:
                # This class is redundant
                remove_class(counts, uncovered_count, m, r)
                classes.pop(i)
                used_moduli.discard(m)
                improved = True
                break # Restart removal pass

        if not improved:
            break

    return classes, counts, uncovered_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    best_classes = None
    best_uncovered = BASE
    best_counts = None
    
    # Initial greedy
    classes, counts, uncovered = greedy_init(randomize=False)
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = classes
        best_counts = counts
    
    # Local search on initial
    classes, counts, uncovered = local_search(classes, counts, uncovered, deadline)
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = classes
        best_counts = counts
        
    # Random restarts
    while time.time() < deadline - 5: # Leave 5s for final return
        # Randomized greedy init
        classes, counts, uncovered = greedy_init(randomize=True)
        
        # Local search
        classes, counts, uncovered = local_search(classes, counts, uncovered, deadline)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = classes
            best_counts = counts
            if best_uncovered == 0:
                break
                
    return best_classes
