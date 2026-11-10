import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Precompute all C(16,3) = 560 triples
    all_triples = list(combinations(range(16), 3))
    
    # Precompute pair-to-triple index for fast overlap checks
    # pair -> list of triples containing that pair
    pair_to_triples = {}
    for t in all_triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (t[i], t[j])
                pair_to_triples.setdefault(pair, []).append(t)
    
    def independence_number(triples):
        """Compute the independence number (largest subset with no triple)."""
        # Use bitmasks for efficiency
        n = 16
        # For each point, precompute which triples contain it
        point_to_triples = [set() for _ in range(n)]
        for idx, t in enumerate(triples):
            for p in t:
                point_to_triples[p].add(idx)
        
        # Build conflict matrix: two points conflict if they share a triple
        # Actually, we need: a subset S is independent if no triple is fully contained in S
        # Use bitmask approach: for each subset, check if any triple is subset of it
        # More efficient: iterate subsets by size from large to small
        
        # Precompute for each triple its bitmask
        triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
        
        # Try from size 8 down (symmetry: if alpha > 8, complement has alpha < 8)
        # Actually just check all sizes from 16 down
        for size in range(16, 0, -1):
            found = False
            for subset in combinations(range(16), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return size
        return 0
    
    def independence_number_fast(triples):
        """Faster independence number using bitmask and early termination."""
        n = 16
        triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
        
        # For each point, bitmask of points that form a triple with it (within each triple)
        # A subset S is independent iff for every triple, not all 3 points are in S
        
        # Use the fact that we can check subsets in decreasing size order
        # and use a clever pruning: if a subset of size k is independent, 
        # all its subsets are also independent, so we only need to check size k.
        
        # Iterate sizes from 16 down to 1
        for size in range(16, 0, -1):
            for subset in combinations(range(16), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained in mask
                is_independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return size
        return 0
    
    def independence_number_bitset(triples):
        """Use bitset operations for speed."""
        n = 16
        triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
        
        for size in range(16, 0, -1):
            for subset in combinations(range(16), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return size
        return 0
    
    def can_add_triple(current_triples, new_triple):
        """Check if new_triple can be added without sharing 2+ points with any existing triple."""
        new_set = set(new_triple)
        for t in current_triples:
            if len(new_set & set(t)) > 1:
                return False
        return True
    
    def greedy_construct(num_triples_target=None, rng=None):
        """Greedy randomized construction."""
        if rng is None:
            rng = random.Random()
        
        triples = []
        used_pairs = set()
        
        available = list(all_triples)
        rng.shuffle(available)
        
        for t in available:
            if num_triples_target and len(triples) >= num_triples_target:
                break
            # Check overlap with all existing triples
            ok = True
            for existing in triples:
                overlap = len(set(t) & set(existing))
                if overlap > 1:
                    ok = False
                    break
            if ok:
                triples.append(t)
        
        return triples
    
    def local_search(triples, rng=None, max_iterations=10000, time_limit=None):
        """Local search: try to swap out a triple for a better one to reduce independence number."""
        if rng is None:
            rng = random.Random()
        
        best_alpha = independence_number_fast(triples)
        best_triples = list(triples)
        
        iteration = 0
        while iteration < max_iterations:
            if time_limit and time.time() > time_limit:
                break
            
            # Randomly pick a triple to remove and try to replace
            idx = rng.randint(0, len(best_triples) - 1)
            removed = best_triples[idx]
            remaining = [t for i, t in enumerate(best_triples) if i != idx]
            
            # Try to add a random triple that doesn't conflict
            candidates = []
            for t in all_triples:
                if t == removed:
                    continue
                ok = True
                for existing in remaining:
                    if len(set(t) & set(existing)) > 1:
                        ok = False
                        break
                if ok:
                    candidates.append(t)
            
            if not candidates:
                iteration += 1
                continue
            
            # Try a few random candidates
            for _ in range(min(5, len(candidates))):
                new_t = rng.choice(candidates)
                new_triples = remaining + [new_t]
                alpha = independence_number_fast(new_triples)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = new_triples
                    break
            
            iteration += 1
            
            if best_alpha <= 5:  # Can't do much better
                break
        
        return best_triples, best_alpha
    
    # Phase 1: Multiple greedy constructions with different seeds
    best_triples = None
    best_alpha = 17  # start high
    
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # leave time for local search
            break
        
        # Vary the number of triples to add
        # More triples = lower independence number but harder to add
        target = rng.randint(20, 50)
        triples = greedy_construct(target, rng)
        
        alpha = independence_number_fast(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(triples)
        
        # Local search on this result
        if time.time() > deadline - 20:
            break
        
        improved_triples, improved_alpha = local_search(
            triples, rng, 
            max_iterations=2000,
            time_limit=deadline - 15
        )
        
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved_triples
    
    # Phase 2: More aggressive local search on the best solution
    if best_triples and time.time() < deadline - 5:
        final_triples, final_alpha = local_search(
            best_triples, rng,
            max_iterations=20000,
            time_limit=deadline - 2
        )
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = final_triples
    
    if best_triples is None:
        # Fallback: return the seed solution
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
