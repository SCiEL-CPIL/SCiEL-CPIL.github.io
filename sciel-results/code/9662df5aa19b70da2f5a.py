import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    N = 16
    ALL_TRIPLES = list(combinations(range(N), 3))
    
    def triple_to_mask(t):
        """Convert a triple (a,b,c) to a bitmask."""
        return (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    def compute_alpha(triple_masks):
        """
        Compute independence number using bitmask enumeration.
        A subset S is independent iff no triple mask is a subset of S.
        We check from largest size down.
        """
        # Precompute: for each point, which triples contain it
        point_to_triples = [[] for _ in range(N)]
        for idx, mask in enumerate(triple_masks):
            for p in range(N):
                if mask & (1 << p):
                    point_to_triples[p].append(idx)
        
        # Check subsets from size N down to 1
        # For efficiency, we can use the fact that if a subset of size k is independent,
        # all its subsets are also independent. So we just need to find the max size.
        
        # Use a smarter approach: iterate over all 2^16 = 65536 subsets
        # and find the largest one with no contained triple
        best = 0
        for s in range(1 << N):
            # Count bits
            bit_count = bin(s).count('1')
            if bit_count <= best:
                continue
            # Check if any triple is fully contained in s
            independent = True
            for mask in triple_masks:
                if (mask & s) == mask:
                    independent = False
                    break
            if independent:
                best = bit_count
                # Can't do better than N, and we're going up
                # Actually we should continue since we might find larger
        return best
    
    def compute_alpha_fast(triple_masks):
        """
        Faster alpha computation using the fact that we only need to check
        subsets that are 'maximal' in some sense. Use a meet-in-the-middle
        or simply enumerate smartly.
        
        Actually, for 16 points, 2^16 = 65536 is fast enough if we're clever.
        Let's precompute which triples are contained in each half.
        """
        if not triple_masks:
            return N
        
        # Split points into two halves of 8
        # For each subset of first half and each subset of second half,
        # check if their union contains any triple.
        
        # Simpler: just iterate all 2^16 subsets but with early termination
        # and bit tricks
        
        # Even simpler: for each possible size k from N down to 1,
        # check if there exists a k-subset with no triple.
        # But that's expensive too.
        
        # Let's use the direct enumeration but optimized:
        best = 0
        for s in range(1, 1 << N):
            bit_count = bin(s).count('1')
            if bit_count <= best:
                continue
            # Quick check: does s contain any triple?
            # We can check all triples
            has_triple = False
            for mask in triple_masks:
                if (mask & s) == mask:
                    has_triple = True
                    break
            if not has_triple:
                best = bit_count
        return best
    
    def is_valid_triple(triple, used_pairs):
        """Check if adding this triple would violate linearity."""
        a, b, c = triple
        pairs = [(a, b), (a, c), (b, c)]
        for p in pairs:
            if p in used_pairs:
                return False
        return True
    
    def greedy_construct(seed_val):
        """Greedy construction with given seed."""
        rng = random.Random(seed_val)
        all_t = list(ALL_TRIPLES)
        rng.shuffle(all_t)
        
        used_pairs = set()
        triples = []
        
        for t in all_t:
            if is_valid_triple(t, used_pairs):
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                triples.append(t)
        
        return triples
    
    def local_search(triples, max_iterations=5000):
        """
        Local search: try to improve by adding, removing, or swapping triples.
        Goal: minimize independence number.
        """
        triples = list(triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
        
        triple_masks = [triple_to_mask(t) for t in triples]
        current_alpha = compute_alpha_fast(triple_masks)
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            
            # Try adding a random triple
            rng = random.Random(iters * 31 + 7)
            candidates = [t for t in ALL_TRIPLES if t not in triples and is_valid_triple(t, used_pairs)]
            
            if candidates:
                # Try a few random additions
                for _ in range(min(5, len(candidates))):
                    t = rng.choice(candidates)
                    # Add it
                    a, b, c = t
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    triples.append(t)
                    triple_masks.append(triple_to_mask(t))
                    
                    new_alpha = compute_alpha_fast(triple_masks)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                    else:
                        # Revert
                        triples.pop()
                        triple_masks.pop()
                        used_pairs.discard((a, b))
                        used_pairs.discard((a, c))
                        used_pairs.discard((b, c))
            
            # Try swapping: remove one triple, add another
            if len(triples) > 0:
                for _ in range(min(3, len(triples))):
                    idx = rng.randrange(len(triples))
                    old_t = triples[idx]
                    a, b, c = old_t
                    used_pairs.discard((a, b))
                    used_pairs.discard((a, c))
                    used_pairs.discard((b, c))
                    triples.pop(idx)
                    triple_masks.pop(idx)
                    
                    # Find a valid triple to add
                    candidates = [t for t in ALL_TRIPLES if t not in triples and is_valid_triple(t, used_pairs)]
                    if candidates:
                        new_t = rng.choice(candidates)
                        a2, b2, c2 = new_t
                        used_pairs.add((a2, b2))
                        used_pairs.add((a2, c2))
                        used_pairs.add((b2, c2))
                        triples.append(new_t)
                        triple_masks.append(triple_to_mask(new_t))
                        
                        new_alpha = compute_alpha_fast(triple_masks)
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            improved = True
                        else:
                            # Revert swap
                            triples.pop()
                            triple_masks.pop()
                            used_pairs.discard((a2, b2))
                            used_pairs.discard((a2, c2))
                            used_pairs.discard((b2, c2))
                            # Restore old triple
                            triples.append(old_t)
                            triple_masks.append(triple_to_mask(old_t))
                            used_pairs.add((a, b))
                            used_pairs.add((a, c))
                            used_pairs.add((b, c))
                    else:
                        # Restore old triple
                        triples.append(old_t)
                        triple_masks.append(triple_to_mask(old_t))
                        used_pairs.add((a, b))
                        used_pairs.add((a, c))
                        used_pairs.add((b, c))
        
        return triples, current_alpha
    
    best_triples = None
    best_indep = N + 1
    
    # Phase 1: Greedy constructions
    num_greedy = 200
    for i in range(num_greedy):
        if time.time() - start_time > DEADLINE:
            break
        
        triples = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        triple_masks = [triple_to_mask(t) for t in triples]
        alpha = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
        
        if best_indep <= 3:
            break
    
    # Phase 2: Local search on best solutions
    if best_triples is not None and time.time() - start_time < DEADLINE:
        # Run local search on the best solution
        improved_triples, improved_alpha = local_search(best_triples, max_iterations=10000)
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved_triples
    
    # Phase 3: More greedy + local search iterations
    for i in range(num_greedy, num_greedy + 300):
        if time.time() - start_time > DEADLINE:
            break
        
        triples = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        triple_masks = [triple_to_mask(t) for t in triples]
        alpha = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
            # Run local search on this promising solution
            if time.time() - start_time < DEADLINE:
                improved_triples, improved_alpha = local_search(best_triples, max_iterations=5000)
                if improved_alpha < best_indep:
                    best_indep = improved_alpha
                    best_triples = improved_triples
        
        if best_indep <= 3:
            break
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return [list(t) for t in best_triples]
