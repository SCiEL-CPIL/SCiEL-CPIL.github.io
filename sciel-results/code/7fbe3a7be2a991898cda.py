import time
import math
import random
import numpy as np

BASE = 51975

def _all_divisors(n):
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(1)
    return sorted(divs)

_DIVISORS = _all_divisors(BASE)

# Precompute masks as numpy uint8 arrays for fast bitwise ops
# Each mask is a bytearray of length ceil(BASE/8) = 6497 bytes
_MASK_BYTES = {}
_MASK_BYTELEN = (BASE + 7) // 8  # 6497

def _mask_to_bytes(mask_int):
    """Convert a Python int bitmask to a bytes object of length _MASK_BYTELEN."""
    return mask_int.to_bytes(_MASK_BYTELEN, 'little')

def _bytes_to_mask(b):
    """Convert bytes to Python int."""
    return int.from_bytes(b, 'little')

# Build masks for each (m, r) pair
# Store as list of (m, r, mask_bytes) tuples for fast access
_ALL_CLASSES = []  # list of (m, r, mask_bytes)
_MASKS_BY_MODULUS = {}  # m -> list of (r, mask_bytes)

for m in _DIVISORS:
    masks_m = []
    for r in range(m):
        mask = 0
        for i in range(r, BASE, m):
            mask |= (1 << i)
        mb = _mask_to_bytes(mask)
        masks_m.append((r, mb))
        _ALL_CLASSES.append((m, r, mb))
    _MASKS_BY_MODULUS[m] = masks_m

def _uncovered_count_bytes(covered_bytes):
    """Count uncovered residues given covered as bytes."""
    # XOR with all-ones to get uncovered, then count bits
    ones = b'\xff' * _MASK_BYTELEN
    uncovered = bytes(a ^ b for a, b in zip(covered_bytes, ones))
    # Count set bits
    return sum(bin(byte).count('1') for byte in uncovered)

def _uncovered_count_int(covered_mask):
    """Count uncovered residues given covered as int."""
    full_mask = (1 << BASE) - 1
    return (~covered_mask & full_mask).bit_count()

def _covered_bytes_from_int(covered_mask):
    return _mask_to_bytes(covered_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Leave 2s margin
    
    best_result = None
    best_score = BASE  # Start with all uncovered
    
    # Use numpy for fast bitwise operations on byte arrays
    # Represent coverage as numpy uint8 array
    
    # Phase 1: Fast greedy using byte-level operations
    # We'll maintain coverage as a Python int for simplicity but use
    # precomputed masks for speed
    
    covered_int = 0
    classes = {}
    available = set(_DIVISORS)
    
    # Precompute for each modulus the list of (r, mask_int) for fast gain computation
    # Actually, let's use the byte representation for speed
    
    # Greedy: repeatedly pick the class with maximum new coverage
    while available and time.time() < deadline - 50:
        best_m = None
        best_r = None
        best_gain = 0
        
        for m in available:
            masks_m = _MASKS_BY_MODULUS[m]
            best_r_m = 0
            best_gain_m = 0
            
            for r, mb in masks_m:
                # Compute gain = popcount(mask & ~covered)
                # Using int for now since numpy on bytes is tricky
                mask_int = _bytes_to_mask(mb)
                gain = (mask_int & ~covered_int).bit_count()
                if gain > best_gain_m:
                    best_gain_m = gain
                    best_r_m = r
            
            if best_gain_m > best_gain:
                best_gain = best_gain_m
                best_m = m
                best_r = best_r_m
        
        if best_m is None or best_gain == 0:
            break
        
        classes[best_m] = best_r
        covered_int |= _bytes_to_mask(_MASKS_BY_MODULUS[best_m][best_r][1])
        available.discard(best_m)
    
    current_uncovered = _uncovered_count_int(covered_int)
    if current_uncovered < best_score:
        best_score = current_uncovered
        best_result = dict(classes)
    
    # Phase 2: Local search - try to improve by changing residues
    if time.time() < deadline - 10:
        improved = True
        max_iter = 50
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            current_uncovered = _uncovered_count_int(covered_int)
            
            # For each modulus in current classes, try to find better residue
            for m in list(classes.keys()):
                if time.time() > deadline - 10:
                    break
                
                # Compute mask of all other classes
                others_mask = 0
                for m2, r2 in classes.items():
                    if m2 != m:
                        others_mask |= _bytes_to_mask(_MASKS_BY_MODULUS[m2][r2][1])
                
                masks_m = _MASKS_BY_MODULUS[m]
                current_r = classes[m]
                
                # Find best residue for this modulus
                best_r = current_r
                best_gain = 0
                for r, mb in masks_m:
                    mask_int = _bytes_to_mask(mb)
                    gain = (mask_int & ~others_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_r != current_r:
                    new_covered = others_mask | _bytes_to_mask(_MASKS_BY_MODULUS[m][best_r][1])
                    new_uncovered = _uncovered_count_int(new_covered)
                    if new_uncovered < current_uncovered:
                        classes[m] = best_r
                        covered_int = new_covered
                        current_uncovered = new_uncovered
                        improved = True
            
            # Try adding a new class if available
            if time.time() < deadline - 10 and len(classes) < len(_DIVISORS):
                best_m_add = None
                best_r_add = None
                best_gain_add = 0
                
                for m in _DIVISORS:
                    if m in classes:
                        continue
                    
                    masks_m = _MASKS_BY_MODULUS[m]
                    best_r_m = 0
                    best_gain_m = 0
                    for r, mb in masks_m:
                        mask_int = _bytes_to_mask(mb)
                        gain = (mask_int & ~covered_int).bit_count()
                        if gain > best_gain_m:
                            best_gain_m = gain
                            best_r_m = r
                    
                    if best_gain_m > best_gain_add:
                        best_gain_add = best_gain_m
                        best_m_add = m
                        best_r_add = best_r_m
                
                if best_m_add is not None and best_gain_add > 0:
                    classes[best_m_add] = best_r_add
                    covered_int |= _bytes_to_mask(_MASKS_BY_MODULUS[best_m_add][best_r_add][1])
                    improved = True
            
            # Try removing a class that doesn't help
            if time.time() < deadline - 10:
                for m in list(classes.keys()):
                    if time.time() > deadline - 10:
                        break
                    
                    others_mask = 0
                    for m2, r2 in classes.items():
                        if m2 != m:
                            others_mask |= _bytes_to_mask(_MASKS_BY_MODULUS[m2][r2][1])
                    
                    without_m_uncovered = _uncovered_count_int(others_mask)
                    with_m_uncovered = _uncovered_count_int(covered_int)
                    
                    if without_m_uncovered <= with_m_uncovered:
                        del classes[m]
                        covered_int = others_mask
                        improved = True
        
        final_uncovered = _uncovered_count_int(covered_int)
        if final_uncovered < best_score:
            best_score = final_uncovered
            best_result = dict(classes)
    
    # Phase 3: Quick simulated annealing if time remains
    if time.time() < deadline - 2:
        temp = 3.0
        cooling = 0.97
        current_uncovered = _uncovered_count_int(covered_int)
        best_sa_uncovered = current_uncovered
        best_sa_classes = dict(classes)
        best_sa_covered = covered_int
        
        sa_deadline = deadline - 2
        while time.time() < sa_deadline:
            move_type = random.random()
            
            if move_type < 0.7 and classes:
                # Change residue of existing class
                m = random.choice(list(classes.keys()))
                old_r = classes[m]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                others_mask = 0
                for m2, r2 in classes.items():
                    if m2 != m:
                        others_mask |= _bytes_to_mask(_MASKS_BY_MODULUS[m2][r2][1])
                
                new_covered = others_mask | _bytes_to_mask(_MASKS_BY_MODULUS[m][new_r][1])
                new_uncovered = _uncovered_count_int(new_covered)
                delta = new_uncovered - current_uncovered
                
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    classes[m] = new_r
                    covered_int = new_covered
                    current_uncovered = new_uncovered
                    
                    if current_uncovered < best_sa_uncovered:
                        best_sa_uncovered = current_uncovered
                        best_sa_classes = dict(classes)
                        best_sa_covered = covered_int
            elif move_type < 0.9 and len(classes) < len(_DIVISORS):
                # Add a class
                avail = [m for m in _DIVISORS if m not in classes]
                if avail:
                    m = random.choice(avail)
                    r = random.randint(0, m - 1)
                    new_covered = covered_int | _bytes_to_mask(_MASKS_BY_MODULUS[m][r][1])
                    new_uncovered = _uncovered_count_int(new_covered)
                    delta = new_uncovered - current_uncovered
                    
                    if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                        classes[m] = r
                        covered_int = new_covered
                        current_uncovered = new_uncovered
                        
                        if current_uncovered < best_sa_uncovered:
                            best_sa_uncovered = current_uncovered
                            best_sa_classes = dict(classes)
                            best_sa_covered = covered_int
            elif move_type < 1.0 and classes:
                # Remove a class
                m = random.choice(list(classes.keys()))
                del classes[m]
                
                new_covered = 0
                for m2, r2 in classes.items():
                    new_covered |= _bytes_to_mask(_MASKS_BY_MODULUS[m2][r2][1])
                new_uncovered = _uncovered_count_int(new_covered)
                delta = new_uncovered - current_uncovered
                
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    covered_int = new_covered
                    current_uncovered = new_uncovered
                    
                    if current_uncovered < best_sa_uncovered:
                        best_sa_uncovered = current_uncovered
                        best_sa_classes = dict(classes)
                        best_sa_covered = covered_int
            
            temp *= cooling
        
        if best_sa_uncovered < best_score:
            best_score = best_sa_uncovered
            best_result = best_sa_classes
    
    # Convert to required format
    if best_result is None:
        return []
    
    result = []
    for m, r in best_result.items():
        result.append([m, r])
        
    return result
