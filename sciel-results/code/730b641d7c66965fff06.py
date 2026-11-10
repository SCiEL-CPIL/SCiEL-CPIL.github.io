"""Rewritten program: construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

def _independence_number(triples):
    """Return the size of the largest subset of {0..15} containing no triple.
    Uses branch-and-bound with bitmasks for speed."""
    n = 16
    # Precompute: for each point, the bitmask of triples that contain it
    # And for each triple, its bitmask of points
    triple_masks = [0] * len(triples)
    for i, (a, b, c) in enumerate(triples):
        triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
    
    # point_to_triples[p] = list of triple indices containing p
    point_to_triples = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(triples):
        point_to_triples[a].append(i)
        point_to_triples[b].append(i)
        point_to_triples[c].append(i)
    
    best = [0]
    
    # We want to find the largest independent set.
    # Independent set: no triple is fully contained.
    # Equivalent: for every triple, at least one point is NOT in the set.
    
    # Branch and bound: try to include or exclude each point.
    # State: current_mask (points chosen so far), remaining points to consider
    # If current_mask already contains a full triple, prune.
    
    def _has_triple(mask):
        for tm in triple_masks:
            if (mask & tm) == tm:
                return True
        return False
    
    # For efficiency, track which triples are still "possible" (not yet fully contained)
    # A triple is "hit" if at least one of its points is NOT chosen.
    # We want to maximize |chosen| such that no triple is fully chosen.
    
    # Alternative: think of it as a hitting set / set cover dual.
    # Let's use a different approach: iterate over all 2^16 subsets? No, that's 65536, feasible!
    
    # Actually 2^16 = 65536, and for each we check if any triple is contained.
    # With 16 triples max (5120 possible triples, but we only have ~5-15 in practice),
    # this is very fast.
    
    max_size = 0
    for mask in range(1 << n):
        # Check if this mask contains any triple
        valid = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                valid = False
                break
        if valid:
            s = bin(mask).count('1')
            if s > max_size:
                max_size = s
                if max_size == n:  # can't do better
                    return max_size
    return max_size


def _are_compatible(triples, new_triple):
    """Check if new_triple can be added to the existing list without violating linearity."""
    a, b, c = new_triple
    mask = (1 << a) | (1 << b) | (1 << c)
    for (x, y, z) in triples:
        existing_mask = (1 << x) | (1 << y) | (1 << z)
        common = mask & existing_mask
        # They share at most one point
        if bin(common).count('1') > 1:
            return False
    return True


def _greedy_construct(seed, time_limit=120):
    """Build a linear hypergraph greedily with randomized ordering, then try to improve."""
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    
    # Generate all 3-subsets of {0..15}
    all_triples = list(itertools.combinations(range(16), 3))
    
    # We want to build a maximal linear system.
    # Strategy: repeatedly pick a random valid triple and add it.
    # To minimize independence number, we want to "cover" points well.
    
    # Better strategy: start with a random ordering of all triples, greedily add
    # if compatible. Then do local search / swap to improve.
    
    best_triples = None
    best_indep = 16  # worst case
    
    num_attempts = 0
    
    while time.time() < deadline - 5:
        num_attempts += 1
        
        # Random shuffle of all triples
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        triples = []
        for t in shuffled:
            if _are_compatible(triples, t):
                triples.append(t)
            # Stop if we've built a reasonably large system
            # A Steiner system S(2,3,16) has 80 blocks, but that's a Steiner triple system
            # which doesn't exist for v=16 (need v≡1 or 3 mod 6; 16≡4 mod 6, so no STS).
            # Max linear 3-uniform on 16 points: each pair in at most one triple.
            # Number of pairs = C(16,2) = 120. Each triple covers 3 pairs.
            # So max triples = 120/3 = 40.
            if len(triples) >= 40:
                break
        
        if not triples:
            continue
        
        # Compute independence number
        indep = _independence_number(triples)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = triples[:]
        
        # If we've achieved a very good score, we can stop
        if best_indep <= 6:
            break
        
        # Periodically check time
        if num_attempts % 50 == 0 and time.time() > deadline - 10:
            break
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_triples


def construct():
    """Construct a linear 3-uniform hypergraph on 16 points minimizing independence number."""
    start = time.time()
    deadline = start + 120  # 120s limit, use 115s to be safe
    
    # Try multiple random seeds
    best_triples = None
    best_indep = 16
    
    # First, try a few quick greedy constructions
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        triples = _greedy_construct(seed, time_limit=min(30, deadline - time.time() - 10))
        if triples:
            indep = _independence_number(triples)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples[:]
    
    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    # Local search: try to swap out triples to reduce independence number
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rng = random.Random(42)
        all_triples = list(itertools.combinations(range(16), 3))
        
        current = best_triples[:]
        current_indep = best_indep
        
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try random swaps: remove one triple, add another
            if len(current) < 2:
                break
            
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current_without = current[:idx] + current[idx+1:]
            
            # Try to find a replacement
            candidates = [t for t in all_triples if t != removed]
            rng.shuffle(candidates)
            
            improved = False
            for new_t in candidates[:50]:  # try up to 50 candidates
                if _are_compatible(current_without, new_t):
                    candidate = current_without + [new_t]
                    # Quick check: does this reduce independence?
                    # Only compute if the new triple covers some previously uncovered structure
                    cand_indep = _independence_number(candidate)
                    if cand_indep < current_indep:
                        current = candidate
                        current_indep = cand_indep
                        best_indep = cand_indep
                        best_triples = current[:]
                        improved = True
                        break
            
            if not improved:
                # Restore
                current = current_without + [removed]
        
        best_triples = current
    
    return best_triples

# EVOLVE-BLOCK-END
