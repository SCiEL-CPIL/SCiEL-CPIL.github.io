"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 12
M = 40
FULL_MASK = (1 << N) - 1
ALL_MASKS = list(range(1 << N))

def _count_comparable_pairs_masks(masks):
    """Count unordered pairs where one mask is strict subset of the other."""
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if mi == mj:
                continue
            if (mi & mj) == mi or (mi & mj) == mj:
                count += 1
    return count

def _masks_to_sets(masks):
    """Convert list of bitmasks to list of lists of ints."""
    result = []
    for mask in masks:
        s = []
        for bit in range(N):
            if mask & (1 << bit):
                s.append(bit)
        result.append(s)
    return result

def _count_pairs(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family[i], family[j]
            if a < b or b < a:
                count += 1
    return count

def _construct():
    n = N
    m = M
    
    deadline = time.time() + 85.0
    random.seed(42)
    
    best_score = -1
    best_family = None
    
    for restart in range(200):
        if time.time() > deadline - 3.0:
            break
        
        # Greedy construction using bitmasks
        family = [0, FULL_MASK]
        used = set(family)
        
        while len(family) < m:
            best_mask = None
            best_gain = -1
            
            # Sample candidates for speed
            sample_size = min(300, (1 << n) - len(used))
            candidates = []
            for _ in range(sample_size):
                c = random.randint(0, (1 << n) - 1)
                if c not in used:
                    candidates.append(c)
            
            for c in candidates:
                gain = 0
                for t in family:
                    if c == t:
                        continue
                    if (c & t) == c or (c & t) == t:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = c
            
            if best_mask is None:
                # Fallback: random
                while True:
                    c = random.randint(0, (1 << n) - 1)
                    if c not in used:
                        best_mask = c
                        break
            
            family.append(best_mask)
            used.add(best_mask)
        
        score = _count_comparable_pairs_masks(family)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        # Local search: exhaustive per-position replacement
        for iteration in range(500):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            
            for i in range(m):
                if time.time() > deadline - 3.0:
                    break
                
                old_mask = family[i]
                best_new_mask = None
                best_new_score = score
                
                # Precompute: for each other element, check comparability
                others = family[:i] + family[i+1:]
                
                # Try all possible replacements (exhaustive over 4096 masks)
                # But that's 4096 * 39 checks per position * 40 positions = ~6.4M per iteration
                # Too slow. Sample instead.
                
                # Sample 200 candidates
                for _ in range(200):
                    candidate = random.randint(0, (1 << n) - 1)
                    if candidate in used:
                        continue
                    
                    # Compute new score efficiently
                    new_score = score
                    # Remove contributions of old_mask with others
                    for t in others:
                        if old_mask != t:
                            if (old_mask & t) == old_mask or (old_mask & t) == t:
                                new_score -= 1
                    # Add contributions of candidate with others
                    for t in others:
                        if candidate != t:
                            if (candidate & t) == candidate or (candidate & t) == t:
                                new_score += 1
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_mask = candidate
                
                if best_new_mask is not None and best_new_mask != old_mask:
                    used.discard(old_mask)
                    used.add(best_new_mask)
                    family[i] = best_new_mask
                    score = best_new_score
                    improved = True
            
            if not improved:
                break
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    # Final polish on best solution
    if best_family is not None:
        family = best_family[:]
        score = _count_comparable_pairs_masks(family)
        used = set(family)
        
        for iteration in range(1000):
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            
            for i in range(m):
                if time.time() > deadline - 1.0:
                    break
                
                old_mask = family[i]
                others = family[:i] + family[i+1:]
                best_new_mask = None
                best_new_score = score
                
                for _ in range(300):
                    candidate = random.randint(0, (1 << n) - 1)
                    if candidate in used:
                        continue
                    
                    new_score = score
                    for t in others:
                        if old_mask != t:
                            if (old_mask & t) == old_mask or (old_mask & t) == t:
                                new_score -= 1
                    for t in others:
                        if candidate != t:
                            if (candidate & t) == candidate or (candidate & t) == t:
                                new_score += 1
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_mask = candidate
                
                if best_new_mask is not None and best_new_mask != old_mask:
                    used.discard(old_mask)
                    used.add(best_new_mask)
                    family[i] = best_new_mask
                    score = best_new_score
                    improved = True
            
            if not improved:
                break
        
        best_family = family
    
    return _masks_to_sets(best_family)

def construct():
    result = _construct()
    # Ensure all elements are Python ints and sequences are lists
    cleaned = []
    for s in result:
        cleaned.append([int(x) for x in s])
    return cleaned
# EVOLVE-BLOCK-END
