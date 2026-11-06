import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    
    def elapsed():
        return time.time() - start_time
    
    def check_pairs_free(triple, used_pairs):
        a, b, c = triple
        return (a, b) not in used_pairs and (a, c) not in used_pairs and (b, c) not in used_pairs
    
    def add_triple(triple, used_pairs):
        a, b, c = triple
        used_pairs.add((a, b))
        used_pairs.add((a, c))
        used_pairs.add((b, c))
    
    def compute_alpha_fast(triple_list):
        """Fast independence number computation using bitmasks."""
        if not triple_list:
            return n
        
        # Represent each triple as a bitmask
        triple_masks = [0] * len(triple_list)
        for i, t in enumerate(triple_list):
            mask = 0
            for x in t:
                mask |= (1 << x)
            triple_masks[i] = mask
        
        nt = len(triple_masks)
        
        # For each subset size from n down to 1, check if any subset of that size
        # contains no triple
        # Use a smarter approach: for each point set, check if it's "triple-free"
        
        # Precompute: for each pair of points, which triples contain both
        # A set S is triple-free if for every triple, S doesn't contain all 3 points
        
        # Better: iterate over subsets in decreasing order, but use pruning
        # For small n=16, we can use a recursive approach with pruning
        
        # Actually, let's use a different approach:
        # The independence number is the maximum k such that there exists a k-set with no triple
        # Equivalent: minimum hitting set of triples has size n - alpha
        
        # For n=16, let's check sizes from 16 down
        # For size k, we need to check if there's a k-subset containing no triple
        
        # Use backtracking with pruning
        points = list(range(n))
        
        def is_triple_free(subset_mask):
            """Check if the set represented by subset_mask contains no complete triple."""
            for tm in triple_masks:
                if (subset_mask & tm) == tm:
                    return False
            return True
        
        # Check from largest to smallest
        for size in range(n, 0, -1):
            if elapsed() > deadline - 5:
                return n  # Return upper bound on timeout
            
            # For large sizes, use complement: a set of size k is triple-free
            # iff its complement (size n-k) is a hitting set
            # For size >= 8, check complement of size <= 8
            if size >= 8:
                comp_size = n - size
                # Check if there's a hitting set of size comp_size
                # A hitting set hits every triple
                found_hitting = False
                for comp in combinations(range(n), comp_size):
                    comp_mask = 0
                    for x in comp:
                        comp_mask |= (1 << x)
                    # Check if comp hits every triple
                    hits_all = True
                    for tm in triple_masks:
                        if (comp_mask & tm) == 0:
                            hits_all = False
                            break
                    if hits_all:
                        found_hitting = True
                        break
                if found_hitting:
                    return size  # There exists a triple-free set of this size
            else:
                # Direct check for small sets
                for subset in combinations(range(n), size):
                    subset_mask = 0
                    for x in subset:
                        subset_mask |= (1 << x)
                    if is_triple_free(subset_mask):
                        return size
        
        return 0
    
    def try_greedy(seed, strategy="random"):
        import random
        rng = random.Random(seed)
        
        used_pairs = set()
        local_triples = []
        
        all_triples = list(combinations(range(n), 3))
        
        if strategy == "random":
            rng.shuffle(all_triples)
        elif strategy == "balanced":
            # Sort by some heuristic
            pass
        elif strategy == "greedy_max":
            # Try to maximize coverage
            pass
        
        for t in all_triples:
            if elapsed() > deadline - 30:
                break
            if check_pairs_free(t, used_pairs):
                local_triples.append(t)
                add_triple(t, used_pairs)
        
        return local_triples
    
    best_triples = []
    best_alpha = n + 1
    
    # Phase 1: Multiple random greedy attempts
    for seed in range(100):
        if elapsed() > deadline - 40:
            break
        triples = try_greedy(seed, "random")
        if triples:
            alpha = compute_alpha_fast(triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
    
    # Phase 2: Try to improve by local search if time allows
    if best_triples and elapsed() < deadline - 20:
        # Try adding more triples or swapping
        used_pairs = set()
        for t in best_triples:
            add_triple(t, used_pairs)
        
        # Try to add more triples
        all_triples = list(combinations(range(n), 3))
        import random
        rng = random.Random(42)
        rng.shuffle(all_triples)
        
        improved = False
        for t in all_triples:
            if elapsed() > deadline - 20:
                break
            if check_pairs_free(t, used_pairs):
                best_triples.append(t)
                add_triple(t, used_pairs)
                improved = True
        
        if improved:
            alpha = compute_alpha_fast(best_triples)
            if alpha < best_alpha:
                best_alpha = alpha
    
    # Fallback
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Verify linearity
    for i in range(len(best_triples)):
        si = set(best_triples[i])
        for j in range(i + 1, len(best_triples)):
            if len(si & set(best_triples[j])) > 1:
                return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
