<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Five disjoint triples.
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _independence_number(triples):
    """Compute the independence number: largest subset of {0..15} containing no triple."""
    n = 16
    # Map each point to the set of triples it belongs to
    # For a subset S, S contains a triple iff some triple is entirely within S
    # We check subsets from largest to smallest
    # Use bitmask: point i -> bit i
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check all subsets from size 16 down to 1
    # A subset S (bitmask) is independent if no triple_mask is a subset of S
    # i.e., for all tm in triple_masks: (tm & S) != tm
    
    # Optimization: for a given S, if any tm is subset of S, S is not independent
    # We want the max |S| such that no tm is subset of S
    
    # Iterate by size descending
    for size in range(n, 0, -1):
        for comb in itertools.combinations(range(n), size):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            # Check if any triple is contained
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _is_valid(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            shared = len(set(triples[i]) & set(triples[j]))
            if shared > 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s budget, leave margin for 150s kill
    
    random.seed(42)
    
    # Initial system: try to build a dense linear triple system
    # Greedy: add triples that maximize coverage
    points = list(range(16))
    triples = []
    
    # Start with a good initial structure
    # Use a greedy approach to pack triples
    all_pairs = set()
    for i in range(16):
        for j in range(i+1, 16):
            all_pairs.add((i, j))
    
    # Greedy construction: repeatedly add a triple that doesn't violate linearity
    # and covers new pairs
    used_pairs = set()
    triples = []
    
    # Try to build a near-maximal linear triple system
    # Use randomized greedy with multiple restarts
    best_triples = None
    best_alpha = 17  # worst case
    
    num_restarts = 50
    triples_per_restart = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Randomized greedy construction
        triples = []
        used_pairs = set()
        
        # Shuffle points for diversity
        point_order = list(range(16))
        random.shuffle(point_order)
        
        # Try to add triples greedily
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            if time.time() > deadline - 30:
                break
            attempts += 1
            
            # Pick 3 random points
            a, b, c = random.sample(range(16), 3)
            triple = (min(a,b,c), min(a,b,c), max(a,b,c))  # wrong, let me fix
            a, b, c = sorted([a, b, c])
            triple = (a, b, c)
            
            pairs = [(a,b), (a,c), (b,c)]
            
            # Check if any pair is already used
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            
            if not conflict:
                triples.append(triple)
                for p in pairs:
                    used_pairs.add(p)
        
        # Compute alpha
        alpha = _independence_number(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(triples)
        
        # Local search: try to improve by replacing triples
        if time.time() > deadline - 15:
            break
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try replacing each triple with a new one
            for idx in range(len(triples)):
                if time.time() > deadline - 15:
                    break
                
                # Remove triple idx
                removed = triples.pop(idx)
                removed_pairs = [(removed[0],removed[1]), (removed[0],removed[2]), (removed[1],removed[2])]
                for p in removed_pairs:
                    used_pairs.discard(p)
                
                # Try to add a new triple
                for _ in range(20):
                    a, b, c = random.sample(range(16), 3)
                    a, b, c = sorted([a, b, c])
                    new_triple = (a, b, c)
                    new_pairs = [(a,b), (a,c), (b,c)]
                    
                    conflict = False
                    for p in new_pairs:
                        if p in used_pairs:
                            conflict = True
                            break
                    
                    if not conflict:
                        triples.append(new_triple)
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        alpha_new = _independence_number(triples)
                        if alpha_new < best_alpha:
                            best_alpha = alpha_new
                            best_triples = list(triples)
                            improved = True
                        else:
                            # Revert
                            triples.pop()
                            for p in new_pairs:
                                used_pairs.discard(p)
                        break  # Only try one replacement per idx per iteration
                
                if time.time() > deadline - 15:
                    break
        
        triples_per_restart += len(triples)
    
    # If we found a good solution, return it
    if best_triples is not None and best_alpha < 11:
        return [list(t) for t in best_triples]
    
    # Fallback: return the initial 5 disjoint triples
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the independence number computation is too slow with the naive approach. Let me optimize it significantly. For 16 points, I can precompute which subsets contain a triple using a more efficient method.

Let me reconsider the approach. The key optimization: for a given set of triples, I can compute for each point which triples it's in, and use a faster independence check.

Actually, for 16 points, 2^16 = 65536 subsets. For each subset, checking if it contains any triple is O(t) where t is the number of triples. With t up to ~40, that's ~2.6M operations per alpha computation. That's fine if I do it a few hundred times.

But the local search loop might do too many alpha computations. Let me be more careful with the time budget and optimize the alpha computation.

A better approach for alpha computation: 
- For each triple, it "covers" all subsets that contain it. 
- We want the largest subset NOT covered by any triple.
- Equivalently: a subset S is independent iff for every triple T, S does not contain T.
- This is equivalent to: S is a subset of the complement of at least one "forbidden pattern"... no, that's not right either.

Actually, the cleanest way: S is independent iff no triple is entirely contained in S. So for each triple T = {a,b,c}, S is NOT independent if a∈S, b∈S, c∈S.

So S is independent iff for all triples {a,b,c}: not (a∈S and b∈S and c∈S).

I can compute this with bitmasks. Let me use a different approach: precompute for each point the list of triples it belongs to, then for a candidate set S, check if there's a triple fully in S.

Actually, the fastest way for n=16: iterate subsets by size descending, and for each subset check if it contains any triple. With early termination, this should be fast enough.

Let me rewrite more carefully with better time management:

