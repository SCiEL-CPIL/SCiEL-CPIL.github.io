"""Linear triple systems on 16 points with small independence number."""
import time
import random
from itertools import combinations

DEADLINE_MARGIN = 5.0  # seconds before the 120s wall-clock limit


def _independence_number(triples):
    """Compute the independence number of a linear triple system.
    
    A subset S of points is independent if no triple is entirely contained in S.
    We find the largest such subset by checking all subsets from largest to smallest.
    """
    n = 16
    # Bitmask representation: triple as a 16-bit integer
    triple_masks = [0] * len(triples)
    for i, t in enumerate(triples):
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[i] = mask
    
    # Check subsets from size 16 down to 1
    # A subset S (as bitmask) is independent if for all triples, S does NOT contain the triple
    # i.e., for all triple_masks tm: (S & tm) != tm
    
    # Generate subsets in order of decreasing size
    # For efficiency, we can use the property that if a subset of size k is independent,
    # we only need to check sizes k+1 and above... but actually we want the LARGEST independent set.
    
    # Brute force: iterate over all 2^16 = 65536 subsets, track max size that's independent
    # But we can be smarter: iterate from size 16 down and stop at first independent set found
    
    # Actually, let's just compute the max over all subsets
    # 2^16 = 65536, for each check against len(triples) triples
    # This is fast enough
    
    max_indep = 0
    num_triples = len(triple_masks)
    
    # Iterate over all subsets, find max size independent set
    # Use Gray code or just iterate all masks
    # For each mask, count bits and check independence
    
    # More efficient: iterate by subset size from 16 down to 1
    # Use itertools.combinations
    
    for size in range(16, 0, -1):
        for combo in combinations(range(16), size):
            mask = 0
            for p in combo:
                mask |= (1 << p)
            # Check if independent: no triple is fully contained
            indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    indep = False
                    break
            if indep:
                return size
    
    return 0


def _can_add(triples, triple_set, mask_count, new_triple):
    """Check if new_triple can be added to the system (linearity: at most 1 shared point)."""
    a, b, c = new_triple
    new_mask = (1 << a) | (1 << b) | (1 << c)
    
    for i, t in enumerate(triples):
        shared = new_mask & mask_count[i]  # This doesn't work as written
        # Actually, let's compute intersection properly
        inter = new_mask & (mask_count[i] if i < len(mask_count) else 0)
        # Wait, I need the actual triple masks
    
    # Let me rewrite this more carefully
    # For each existing triple, check if they share at most 1 point
    for t in triples:
        shared = 0
        if a in t:
            shared += 1
        if b in t:
            shared += 1
        if c in t:
            shared += 1
        if shared > 1:
            return False
    return True


def _greedy_construct(random_seed=None):
    """Construct a linear triple system using greedy + local search."""
    if random_seed is not None:
        random.seed(random_seed)
    
    n = 16
    all_points = list(range(n))
    all_triples = list(combinations(range(n), 3))
    
    # Start with a random valid triple
    random.shuffle(all_triples)
    
    triples = []
    triple_masks = []
    
    # Greedily add triples
    for t in all_triples:
        a, b, c = t
        new_mask = (1 << a) | (1 << b) | (1 << c)
        
        can_add = True
        for i, existing in enumerate(triples):
            existing_mask = triple_masks[i]
            shared = bin(new_mask & existing_mask).count('1')
            if shared > 1:
                can_add = False
                break
        
        if can_add:
            triples.append(list(t))
            triple_masks.append(new_mask)
    
    # Local search: try to swap out a triple for a better one
    # We want to minimize the independence number
    
    best_indep = _independence_number(triples)
    
    # Try random mutations
    for _ in range(5000):
        if time.time() - start_time > 110:
            break
        
        # Pick a random triple to remove
        idx = random.randint(0, len(triples) - 1)
        removed = triples.pop(idx)
        removed_mask = triple_masks.pop(idx)
        
        # Try to find a replacement triple
        found = False
        for t in random.sample(all_triples, min(100, len(all_triples))):
            a, b, c = t
            new_mask = (1 << a) | (1 << b) | (1 << c)
            
            can_add = True
            for i, existing in enumerate(triples):
                shared = bin(new_mask & triple_masks[i]).count('1')
                if shared > 1:
                    can_add = False
                    break
            
            if can_add:
                triples.append(list(t))
                triple_masks.append(new_mask)
                found = True
                break
        
        if found:
            new_indep = _independence_number(triples)
            if new_indep < best_indep:
                best_indep = new_indep
        else:
            # Restore removed triple
            triples.insert(idx, removed)
            triple_masks.insert(idx, removed_mask)
    
    return triples, best_indep


start_time = time.time()

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    global start_time
    start_time = time.time()
    
    best_triples = None
    best_indep = 16  # worst case
    
    # Multiple random restarts
    for seed in range(100):
        elapsed = time.time() - start_time
        if elapsed > 110:
            break
        
        triples, indep = _greedy_construct(random_seed=seed)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
        
        if best_indep <= 7:  # Can't do much better than this likely
            break
    
    # If we haven't found good ones, use a deterministic construction
    if best_triples is None:
        # Fallback: the original 5 disjoint triples
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
