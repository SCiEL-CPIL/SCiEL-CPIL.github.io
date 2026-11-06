"""Search for linear triple systems on 16 points with small independence number."""
import time
import random
import sys
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _check_linearity(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _independence_number(triples):
    """Find the largest subset of points containing no complete triple."""
    points = set()
    for t in triples:
        points.update(t)
    n = len(points)
    if n == 0:
        return 0
    triple_set = set()
    for t in triples:
        triple_set.add(frozenset(t))
    
    # Check all subsets from largest to smallest
    # For n=16, check subsets of size 15, 14, ...
    all_points = sorted(points)
    for size in range(n, 0, -1):
        for subset in combinations(all_points, size):
            subset_set = set(subset)
            has_triple = False
            for t in triples:
                if set(t) <= subset_set:
                    has_triple = True
                    break
            if not has_triple:
                return size
    return 0


def _independence_number_fast(triples):
    """
    Calculate independence number using bitmasks.
    Optimized for n=16.
    """
    if not triples:
        return 16
        
    # Map points to 0..15 (assume all 16 points are used or we map to 0..15 directly)
    # We assume points are 0..15.
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
        
    # To find the max independent set, we iterate masks from (1<<16)-1 down to 0.
    # We stop at the first mask that does not contain any triple.
    # This is O(2^16 * T) in worst case, but usually finds high alpha quickly?
    # No, high alpha means the mask is large. 
    # Actually, we want to MINIMIZE the independence number.
    # The evaluator returns the independence number.
    # So we need to compute it accurately.
    
    # A faster way for n=16:
    # Iterate all 2^16 masks.
    # For each mask, check if it is independent.
    # Keep track of the max popcount of an independent mask.
    
    # Optimization: 
    # If a mask is independent, all its submasks are independent.
    # If a mask is NOT independent, all its supermasks are NOT independent.
    # We want the max popcount of an independent mask.
    
    # We can iterate masks in descending order of popcount? 
    # Generating all 65536 masks and sorting by popcount is expensive.
    
    # Better: Iterate masks from 0 to 65535.
    # Maintain an array `is_independent`? No, memory is fine but time?
    
    # Let's use the property: 
    # For a fixed triple mask T, a mask M is "bad" if (M & T) == T.
    # We want to find M with max popcount such that for all T in triple_masks, (M & T) != T.
    
    # Simple loop:
    max_alpha = 0
    for mask in range(1 << 16):
        # Check if mask is independent
        independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                independent = False
                break
        if independent:
            # Count bits
            # bin(mask).count("1") is slow.
            # Use int.bit_count() (Python 3.10+) or fallback
            try:
                pop = mask.bit_count()
            except AttributeError:
                pop = bin(mask).count('1')
            if pop > max_alpha:
                max_alpha = pop
                if max_alpha == 16:
                    break
    return max_alpha


def _try_complete(triples, all_triples, time_limit):
    """Try to extend partial triples to a full system."""
    start = time.time()
    triples = [list(t) for t in triples]
    
    def backtrack(idx, used_pairs):
        if time.time() > time_limit:
            return None
        if idx == len(all_triples):
            return triples[:]
        
        tri = all_triples[idx]
        # Check if this triple conflicts
        tri_set = set(tri)
        for existing in triples:
            common = tri_set & set(existing)
            if len(common) > 1:
                return None
        
        triples.append(tri)
        result = backtrack(idx + 1, used_pairs)
        if result is not None:
            return result
        triples.pop()
        return None
    
    return backtrack(0, set())


def _greedy_construct(random_state):
    """Construct a linear triple system using greedy + local search."""
    random.seed(random_state)
    points = list(range(16))
    
    # Generate all possible triples
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    triples = []
    for tri in all_triples:
        valid = True
        for existing in triples:
            if len(set(tri) & set(existing)) > 1:
                valid = False
                break
        if valid:
            triples.append(list(tri))
        if len(triples) >= 56:  # Steiner triple system has 56 triples for v=16? No, (16*15)/6 = 40
            break
    
    return triples


def _local_search(triples, random_state, time_limit):
    """Improve the system by trying to reduce independence number."""
    random.seed(random_state)
    best_triples = [list(t) for t in triples]
    best_score = _independence_number_fast(best_triples)
    
    # Precompute all possible triples for fast access
    all_triples = list(combinations(range(16), 3))
    
    while time.time() < time_limit:
        if len(best_triples) == 0:
            break
            
        idx = random.randint(0, len(best_triples) - 1)
        old_tri = best_triples[idx]
        old_mask = 0
        for p in old_tri:
            old_mask |= (1 << p)
            
        # Find a random valid replacement
        # We want to swap old_tri with a new_tri that doesn't conflict with others
        # Conflict check: (new_mask & existing_mask) has popcount <= 1
        
        # To speed up, we can just pick a random candidate from all_triples
        # and check validity.
        
        # Optimization: Pick a random triple from all_triples
        new_tri = random.choice(all_triples)
        new_mask = 0
        for p in new_tri:
            new_mask |= (1 << p)
            
        # Check if new_tri conflicts with any existing triple (excluding old_tri)
        valid = True
        for j, existing in enumerate(best_triples):
            if j == idx:
                continue
            # Check intersection size
            # Since triples are small, set intersection is fast enough
            # Or use bitmasks
            if (new_mask & (1 << existing[0])) and (new_mask & (1 << existing[1])) and (new_mask & (1 << existing[2])):
                # This check is wrong, it checks if all 3 are in new_mask.
                # We need intersection size > 1.
                pass
            
            # Simpler: 
            common = set(new_tri) & set(existing)
            if len(common) > 1:
                valid = False
                break
                
        if valid:
            best_triples[idx] = new_tri
            score = _independence_number_fast(best_triples)
            if score < best_score:
                best_score = score
            else:
                best_triples[idx] = old_tri
                
    return best_triples, best_score


def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    time_limit = start_time + 115.0  # 5s safety margin before 120s kill
    
    best_triples = None
    best_score = 16  # worst case
    
    # Phase 1: Multiple greedy constructions with different random seeds
    # We need to run many iterations because the independence number calculation is now fast
    seed = 0
    while time.time() < time_limit - 5:  # Leave 5s for local search
        triples = _greedy_construct(seed)
        if not triples:
            seed += 1
            continue
        score = _independence_number_fast(triples)
        if score < best_score:
            best_score = score
            best_triples = [list(t) for t in triples]
        seed += 1
        
    # Phase 2: Local search to improve
    if best_triples is not None:
        remaining_time = time_limit - time.time()
        if remaining_time > 2:
            # Run local search
            triples, score = _local_search(best_triples, 42, time_limit)
            if score < best_score:
                best_score = score
                best_triples = triples
    
    if best_triples is not None:
        return best_triples
    
    # Fallback: seed program
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
