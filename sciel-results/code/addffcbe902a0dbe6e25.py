import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    N = 16

    def time_check():
        return time.time() - start_time > DEADLINE

    def is_linear(triples):
        """Check if any two triples share more than 1 point."""
        seen_pairs = set()
        for t in triples:
            a, b, c = t
            pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
            for p in pairs:
                if p in seen_pairs:
                    return False
                seen_pairs.add(p)
        return True

    def independence_number(triples):
        """Compute the largest subset of {0..15} containing no triple.
        Uses bitmask brute force over 2^16 = 65536 subsets."""
        if not triples:
            return N
        # Precompute for each point, the bitmask of triples it belongs to
        # Actually, we need: for each triple, its bitmask. Then for each candidate subset,
        # check if any triple is fully contained.
        
        # Fast approach: for each subset mask, check if it contains any triple.
        # Precompute triple_masks: list of 16-bit masks
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        # For each mask from full down to 0, check if it's independent
        # A mask is independent if no triple_mask is a subset of mask
        # i.e., for all tm in triple_masks: (mask & tm) != tm
        
        best = 0
        # Iterate masks in order of decreasing popcount? Or just check all.
        # 65536 masks * len(triples) checks. If len(triples) ~ 56, that's ~3.6M ops. Fast.
        
        # Actually, we can be smarter: iterate masks from (1<<16)-1 down to 0,
        # and track the maximum popcount among independent sets.
        # But that's still 65536 iterations.
        
        # Even smarter: for each mask, if it contains a triple, all its supersets also contain it.
        # But checking is the bottleneck.
        
        # Let's just iterate all masks and check independence.
        # To speed up, we can precompute for each point the list of triples containing it.
        # But the direct triple_mask check is probably fast enough.
        
        n_tm = len(triple_masks)
        best_indep = 0
        
        for mask in range((1 << N) - 1, -1, -1):
            popc = bin(mask).count('1')
            if popc <= best_indep:
                continue
            # Check if mask is independent
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                best_indep = popc
                if best_indep >= 8:  # Can't do better than what we're searching for
                    break
        
        return best_indep

    def compute_alpha_fast(triples):
        """Faster alpha using precomputed structure."""
        if not triples:
            return N
        
        # For each point, which triples does it belong to?
        point_triples = [[] for _ in range(N)]
        for idx, t in enumerate(triples):
            for p in t:
                point_triples[p].append(idx)
        
        # A subset S is independent iff for every triple, not all 3 points are in S.
        # Equivalent: the complement of S is a hitting set.
        # So alpha = N - minimum hitting set size.
        # Find minimum hitting set via brute force over subsets of size 1, 2, 3, ...
        
        # Precompute triple masks
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        n_tm = len(triple_masks)
        
        # Check if there's a hitting set of size k for k=1,2,3,...
        # Hitting set H: for every triple, H intersects it.
        # Equivalent: complement S = V\H has no triple fully contained.
        
        # For small k, enumerate all subsets of size k.
        all_points = range(N)
        
        for k in range(1, N + 1):
            # Check all subsets of size k
            for subset in combinations(all_points, k):
                # Build mask of subset
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if this is a hitting set: for every triple, it intersects subset
                # i.e., for every tm, (mask & tm) != 0
                is_hitting = True
                for tm in triple_masks:
                    if (mask & tm) == 0:
                        is_hitting = False
                        break
                if is_hitting:
                    # Hitting set of size k found, so alpha <= N - k
                    return N - k
        
        return 0

    def alpha_by_complement(triples):
        """Find alpha by searching for minimum hitting set."""
        if not triples:
            return N
        
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        n_tm = len(triple_masks)
        all_points = list(range(N))
        
        for k in range(1, N + 1):
            for subset in combinations(all_points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                is_hitting = True
                for tm in triple_masks:
                    if (mask & tm) == 0:
                        is_hitting = False
                        break
                if is_hitting:
                    return N - k
        
        return 0

    def greedy_construct(seed):
        """Greedy construction of a linear triple system."""
        rng = random.Random(seed)
        all_triples = list(combinations(range(N), 3))
        rng.shuffle(all_triples)
        
        used_pairs = set()
        triples = []
        
        for t in all_triples:
            a, b, c = t
            pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples.append(list(t))
                for p in pairs:
                    used_pairs.add(p)
        
        return triples

    def local_search(triples, max_iters=500):
        """Local search: try to reduce independence number by adding/removing/swapping triples."""
        if not triples:
            return triples
        
        current = list(triples)
        current_alpha = alpha_by_complement(current)
        
        for iteration in range(max_iters):
            if time_check():
                break
            
            # Try adding a random valid triple
            all_triples = list(combinations(range(N), 3))
            rng = random.Random(iteration * 1000 + 42)
            rng.shuffle(all_triples)
            
            improved = False
            for t in all_triples:
                a, b, c = t
                pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
                
                # Check if adding t maintains linearity
                valid = True
                for existing in current:
                    ea, eb, ec = existing
                    epairs = frozenset((ea, eb)), frozenset((ea, ec)), frozenset((eb, ec))
                    for p in pairs:
                        if p in epairs:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_triples = current + [list(t)]
                new_alpha = alpha_by_complement(new_triples)
                
                if new_alpha < current_alpha:
                    current = new_triples
                    current_alpha = new_alpha
                    improved = True
                    break
            
            if not improved:
                # Try removing a triple that doesn't increase alpha (or removes one that's "useless")
                # Or try swapping
                # For now, just stop if no improvement from adding
                break
        
        return current

    # Main search
    best_triples = None
    best_alpha = N
    
    # Phase 1: Many greedy restarts
    num_greedy = 200
    for i in range(num_greedy):
        if time_check():
            break
        triples = greedy_construct(i * 104729 + 7)
        if not triples:
            continue
        alpha = alpha_by_complement(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
        if best_alpha <= 5:
            break
    
    # Phase 2: Local search on best and some others
    if best_triples is not None:
        improved = local_search(best_triples, max_iters=300)
        imp_alpha = alpha_by_complement(improved)
        if imp_alpha < best_alpha:
            best_alpha = imp_alpha
            best_triples = improved
    
    # Phase 3: More greedy with local search
    for i in range(num_greedy, num_greedy + 100):
        if time_check():
            break
        triples = greedy_construct(i * 104729 + 7)
        if not triples:
            continue
        alpha = alpha_by_complement(triples)
        if alpha < best_alpha:
            improved = local_search(triples, max_iters=200)
            imp_alpha = alpha_by_complement(improved)
            if imp_alpha < best_alpha:
                best_alpha = imp_alpha
                best_triples = improved
            else:
                best_alpha = alpha
                best_triples = triples
        elif alpha == best_alpha and best_triples is not None:
            # Try local search even on ties
            improved = local_search(triples, max_iters=100)
            imp_alpha = alpha_by_complement(improved)
            if imp_alpha < best_alpha:
                best_alpha = imp_alpha
                best_triples = improved
        
        if best_alpha <= 5:
            break
    
    if best_triples is None:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
