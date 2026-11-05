import random
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Points 0..15
    points = list(range(16))
    
    # Helper to check linearity of a list of triples
    def is_linear(triples):
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                if len(set(triples[i]) & set(triples[j])) > 1:
                    return False
        return True

    # Helper to calculate alpha (independence number)
    # Alpha is the size of the largest subset of points containing no triple.
    # We iterate all 2^16 subsets.
    def calc_alpha(triples):
        if not triples:
            return 16
        
        # Pre-compute triple masks for fast intersection check?
        # Or just check membership.
        # A subset S is independent if no triple is a subset of S.
        # Equivalently, for all t in triples, t is not subset of S.
        
        max_alpha = 0
        # Iterate all subsets
        for mask in range(1 << 16):
            # Check if mask is independent
            is_independent = True
            for t in triples:
                # Check if t is subset of mask
                # t is a tuple of 3 points
                # (mask >> p) & 1 for each p in t
                if ((mask >> t[0]) & 1) and ((mask >> t[1]) & 1) and ((mask >> t[2]) & 1):
                    is_independent = False
                    break
            
            if is_independent:
                # Count bits
                # bin(mask).count('1') is slow, use bit_count if available (Python 3.10+)
                # Fallback:
                # count = 0
                # m = mask
                # while m:
                #     count += 1
                #     m &= m - 1
                # But bin is likely fast enough for 65k iterations
                size = bin(mask).count('1')
                if size > max_alpha:
                    max_alpha = size
                    # Early exit if we reach 16? No, max is 16.
                    # If max_alpha == 16, we can break? No, we want min alpha, so we are checking a specific config.
        
        return max_alpha

    # Initial construction: Greedy packing
    # Try to add triples that are "good"
    # We'll use a random restart approach with local search.
    
    best_alpha = 16
    best_triples = []
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    random.shuffle(all_triples)
    
    # Helper to check if a triple can be added to a set
    def can_add(triples, new_t):
        t_set = set(new_t)
        for t in triples:
            if len(t_set & set(t)) > 1:
                return False
        return True

    # Simulated Annealing / Hill Climbing
    # We will try to start with a random valid set and improve.
    
    def random_initial():
        triples = []
        available = all_triples[:]
        random.shuffle(available)
        for t in available:
            if can_add(triples, t):
                triples.append(t)
            if len(triples) > 40: # Cap size to avoid huge alpha? No, more triples usually lower alpha.
                # Actually, adding triples can only decrease or keep alpha same.
                # So we want as many as possible?
                # Not necessarily. Some triples might be "bad" if they don't cover new points efficiently.
                # But generally, more constraints = smaller independent set.
                pass
        return triples

    # Local Search Step
    def local_search(triples):
        current_alpha = calc_alpha(triples)
        improved = True
        while improved:
            improved = False
            # Try removing a triple? (Usually bad, increases alpha)
            # Try swapping?
            # Try adding a triple?
            
            # 1. Try adding a random triple
            candidates = [t for t in all_triples if can_add(triples, t)]
            if candidates:
                # Pick one that minimizes alpha
                # To save time, just try a few
                random.shuffle(candidates)
                best_t = None
                best_a = current_alpha
                for t in candidates[:10]: # Limit check
                    new_triples = triples + [t]
                    a = calc_alpha(new_triples)
                    if a < best_a:
                        best_a = a
                        best_t = t
                if best_t is not None:
                    triples.append(best_t)
                    current_alpha = best_a
                    improved = True
                    continue
            
            # 2. Try swapping: remove one, add one
            # This is expensive. 
            # Instead, let's just rely on the add step and maybe a "kick" step.
            
        return triples, current_alpha

    # Main loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Random restart or continue from best?
        # Let's do random restarts with local search.
        init_triples = random_initial()
        improved_triples, alpha = local_search(init_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples
            # Print for debugging? No.
            # If we hit theoretical lower bound, break?
            # Lower bound for alpha:
            # If we have a Steiner system, alpha is small.
            # For v=16, min alpha is likely around 5-7.
            if best_alpha <= 5:
                break

    # If best_triples is empty (should not happen), return seed
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
