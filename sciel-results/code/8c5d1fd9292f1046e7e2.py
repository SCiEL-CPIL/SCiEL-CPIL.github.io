"""Optimized covering system search for moduli dividing 51975."""
import time
import random
import math

BASE = 51975

def _get_divisors():
    """Get all divisors of BASE greater than 1."""
    divisors = []
    for i in range(2, BASE + 1):
        if BASE % i == 0:
            divisors.append(i)
    return divisors

def _precompute_masks(divisors):
    """Precompute bitmasks for each divisor's residue classes."""
    # For each divisor m, precompute the list of positions for each residue r
    # positions[m][r] = list of indices i where i % m == r
    positions = {}
    for m in divisors:
        pos = [[] for _ in range(m)]
        for i in range(BASE):
            pos[i % m].append(i)
        positions[m] = pos
    return positions

def _count_uncovered(covered, positions, m, r):
    """Count how many uncovered residues would be covered by adding (m, r)."""
    pos = positions[m][r]
    return sum(1 for i in pos if not covered[i])

def _greedy_build(divisors, positions, order='ascending'):
    """Greedy construction: pick moduli in given order, best residue for each."""
    covered = bytearray(BASE)
    classes = []
    divs = sorted(divisors) if order == 'ascending' else sorted(divisors, reverse=True)
    for m in divs:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = sum(1 for i in positions[m][r] if not covered[i])
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        for i in positions[m][best_r]:
            covered[i] = 1
    return classes, covered

def _score(covered):
    """Count uncovered residues."""
    return BASE - sum(covered)

def _local_search(classes, positions, max_time=30.0):
    """Local search: try swapping residues or adding/removing moduli."""
    covered = bytearray(BASE)
    for m, r in classes:
        for i in positions[m][r]:
            covered[i] = 1
    current_score = _score(covered)
    best_classes = classes[:]
    best_score = current_score
    
    start = time.time()
    deadline = start + max_time
    
    # Build set of used moduli for quick lookup
    used_moduli = set(m for m, r in classes)
    all_divs = sorted(positions.keys())
    
    # Local search: try changing residue for each modulus
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx, (m, r) in enumerate(classes):
            # Try all other residues for this modulus
            for new_r in range(m):
                if new_r == r:
                    continue
                # Calculate marginal gain: what we lose and what we gain
                # Remove old class
                old_pos = positions[m][r]
                new_pos = positions[m][new_r]
                
                # Count how many positions in old_pos are only covered by this class
                # (i.e., no other class covers them)
                # This is expensive, so we do it efficiently
                # Temporarily remove and check
                # Actually, let's compute the delta directly
                
                # Positions covered by old but not new
                old_set = set(old_pos)
                new_set = set(new_pos)
                
                # For positions in old but not new: check if covered by other classes
                # For positions in new but not old: they become covered
                
                # This is getting complex. Let's use a different approach:
                # Just try the swap and measure the score change
                # Remove old
                for i in old_pos:
                    covered[i] = 0
                # Add new
                for i in new_pos:
                    covered[i] = 1
                
                new_score = _score(covered)
                
                if new_score < current_score:
                    # Accept
                    classes[idx] = [m, new_r]
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                    improved = True
                else:
                    # Revert
                    for i in old_pos:
                        covered[i] = 1
                    for i in new_pos:
                        covered[i] = 0
        
        # Also try adding a new modulus
        if time.time() < deadline:
            for m in all_divs:
                if m in used_moduli:
                    continue
                best_r = 0
                best_gain = -1
                for r in range(m):
                    gain = sum(1 for i in positions[m][r] if not covered[i])
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                if best_gain > 0:
                    # Try adding
                    classes.append([m, best_r])
                    for i in positions[m][best_r]:
                        covered[i] = 1
                    new_score = _score(covered)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_classes = [c[:] for c in classes]
                        improved = True
                    else:
                        # Revert
                        classes.pop()
                        for i in positions[m][best_r]:
                            covered[i] = 0
        
        # Try removing a modulus
        if time.time() < deadline:
            for idx in range(len(classes) - 1, -1, -1):
                m, r = classes[idx]
                # Check if removing this class increases uncovered count
                # Count positions in this class that are only covered by this class
                pos = positions[m][r]
                # For each position, check if it's covered by any other class
                removable = True
                for i in pos:
                    # Check if covered by other classes
                    is_covered_elsewhere = False
                    for j, (m2, r2) in enumerate(classes):
                        if j == idx:
                            continue
                        if i % m2 == r2:
                            is_covered_elsewhere = True
                            break
                    if not is_covered_elsewhere:
                        removable = False
                        break
                if removable:
                    # Remove it
                    for i in pos:
                        covered[i] = 0
                    classes.pop(idx)
                    new_score = _score(covered)
                    if new_score <= current_score:
                        if new_score < best_score:
                            best_score = new_score
                            best_classes = [c[:] for c in classes]
                        improved = True
                    else:
                        # Revert
                        classes.insert(idx, [m, r])
                        for i in pos:
                            covered[i] = 1
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    divisors = _get_divisors()
    positions = _precompute_masks(divisors)
    
    # Try multiple initializations
    best_classes = None
    best_score = BASE  # Worst case: nothing covered
    
    # Ascending greedy
    classes1, covered1 = _greedy_build(divisors, positions, 'ascending')
    score1 = _score(covered1)
    if score1 < best_score:
        best_score = score1
        best_classes = classes1[:]
    
    # Descending greedy
    classes2, covered2 = _greedy_build(divisors, positions, 'descending')
    score2 = _score(covered2)
    if score2 < best_score:
        best_score = score2
        best_classes = classes2[:]
    
    # Local search from best initial solution
    if time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        best_classes, best_score = _local_search(best_classes, positions, max_time=remaining_time)
    
    return best_classes
