import itertools
import random
import time

def construct():
    n = 16
    # Safety margin: 120s limit, use 115s
    deadline = time.time() + 115.0
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pairs for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset([(a, b), (a, c), (b, c)])
        
    # Precompute bitmasks for triples
    triple_masks = {}
    for t in all_triples:
        a, b, c = t
        triple_masks[t] = (1 << a) | (1 << b) | (1 << c)
        
    # Precompute masks for all subsets of size k?
    # To speed up alpha calculation, we can precompute lists of masks for each k
    subset_masks = {}
    for k in range(1, n + 1):
        masks = []
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks.append(mask)
        subset_masks[k] = masks

    def check_alpha_le_k(triples, k):
        """
        Returns True if the independence number is <= k.
        This is equivalent to: every subset of size k+1 contains at least one triple.
        """
        if k >= n:
            return True
        if k < 0:
            return False
            
        # We need to check all subsets of size k+1
        target_k = k + 1
        if target_k > n:
            return True
            
        masks_to_check = subset_masks[target_k]
        triple_m = [triple_masks[t] for t in triples]
        
        for mask in masks_to_check:
            # Check if any triple is subset of mask
            # (t_mask & mask) == t_mask
            is_covered = False
            for tm in triple_m:
                if (tm & mask) == tm:
                    is_covered = True
                    break
            if not is_covered:
                return False
        return True

    def calculate_alpha(triples):
        """
        Calculate the exact independence number.
        """
        # Binary search or linear search from 16 down
        # Since n=16, linear is fine.
        for k in range(16, 0, -1):
            # If alpha <= k is False, then alpha > k
            # We want the largest k such that alpha <= k is True?
            # No, alpha is the size of the LARGEST independent set.
            # So if there is an independent set of size k, alpha >= k.
            # check_alpha_le_k checks if there is NO independent set of size k+1.
            # So if check_alpha_le_k(k) is True, then alpha <= k.
            # We want the smallest k such that check_alpha_le_k(k) is True?
            # No, alpha is the max size.
            # If check_alpha_le_k(5) is True, alpha <= 5.
            # If check_alpha_le_k(4) is True, alpha <= 4.
            # We want to find the actual alpha.
            # So we check k=16, 15, ...
            # If check_alpha_le_k(k) is True, it means alpha <= k.
            # The first k (starting from 16 down) for which this is True is the alpha?
            # Wait.
            # If alpha = 5.
            # check_alpha_le_k(16) -> True (alpha <= 16)
            # check_alpha_le_k(15) -> True
            # ...
            # check_alpha_le_k(5) -> True (alpha <= 5)
            # check_alpha_le_k(4) -> False (alpha > 4)
            
            # So we can just return the smallest k where check_alpha_le_k(k) is True?
            # No, we iterate from 16 down.
            # If we find check_alpha_le_k(k) is False, then alpha > k.
            # We want the largest independent set.
            # So if check_alpha_le_k(k) is True, alpha is at most k.
            # If we iterate k from 16 down to 1:
            # The first k where check_alpha_le_k(k) is True is an upper bound.
            # But we want the tightest bound.
            # Actually, if check_alpha_le_k(k) is True, it doesn't mean alpha == k.
            # It means alpha <= k.
            # To find exact alpha, we can find the largest k such that there EXISTS an independent set of size k.
            # This is equivalent to: check_alpha_le_k(k-1) is False.
            
            # Let's just iterate k from 1 to 16 and find the max k where an independent set exists.
            # An independent set of size k exists iff check_alpha_le_k(k-1) is False.
            
            pass
        
        # Simpler approach:
        # Iterate k from 16 down to 1.
        # If there is an independent set of size k, return k.
        # How to check if there is an independent set of size k?
        # Iterate all subsets of size k. If any is independent, return k.
        
        triple_m = [triple_masks[t] for t in triples]
        
        for k in range(16, 0, -1):
            masks = subset_masks[k]
            for mask in masks:
                is_independent = True
                for tm in triple_m:
                    if (tm & mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 1

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    def generate_random_packing():
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        rng.shuffle(candidates)
        
        for t in candidates:
            ps = triple_pairs[t]
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        
        return triples, used_pairs

    def local_search(triples, used_pairs, current_alpha, time_limit):
        nonlocal best_alpha, best_triples
        
        while time.time() < time_limit:
            if not triples:
                break
                
            idx = rng.randrange(len(triples))
            t_old = triples[idx]
            ps_old = triple_pairs[t_old]
            
            # Remove t_old
            used_pairs.difference_update(ps_old)
            
            # Find valid replacement
            # Shuffle candidates to avoid bias
            candidates = all_triples[:]
            rng.shuffle(candidates)
            
            found = False
            for t_new in candidates:
                if t_new == t_old:
                    continue
                ps_new = triple_pairs[t_new]
                if not (ps_new & used_pairs):
                    # Valid
                    triples[idx] = t_new
                    used_pairs.update(ps_new)
                    
                    new_alpha = calculate_alpha(triples)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                        
                    # Accept move if it doesn't increase alpha (or even if it does, with prob?)
                    # For simplicity, accept if new_alpha <= current_alpha
                    if new_alpha <= current_alpha:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        used_pairs.difference_update(ps_new)
                        triples[idx] = t_old
                        used_pairs.update(ps_old)
                        
                    found = True
                    break
            
            if not found:
                # Revert
                used_pairs.update(ps_old)
                triples[idx] = t_old
                
            if best_alpha <= 5:
                # Likely optimal or near optimal
                return

    # Main Loop
    while time.time() < deadline:
        triples, used_pairs = generate_random_packing()
        current_alpha = calculate_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        # Run local search for a short duration
        search_deadline = min(time.time() + 0.5, deadline - 2.0)
        if search_deadline > time.time():
            local_search(triples, used_pairs, current_alpha, search_deadline)
            
        if best_alpha <= 5:
            break

    return best_triples
