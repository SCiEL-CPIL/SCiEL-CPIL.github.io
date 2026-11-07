import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55
    
    # Precompute comparability bitsets for faster gain calculations
    # comp_mask[s] = bitmask of all t such that s and t are comparable (including s itself)
    # Actually, let's use a different approach: precompute for each mask, the list of comparable masks
    # But that's 4096 * up to 4096 entries, too much.
    # Instead, use the bitwise check but optimize the inner loop.
    
    # Precompute: for each mask, a sorted list of masks comparable to it? Too expensive.
    # Let's just use the bitwise check but make the SA faster.
    
    def is_comparable(a, b):
        # Fast check: a is subset of b OR b is subset of a
        return (a | b) == a or (a | b) == b
    
    def run_single_attempt(seed, time_budget):
        random.seed(seed)
        start_time = time.time()
        
        family = []
        fam_set = set()
        
        # Start with the full chain
        full_chain = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
        for mask in full_chain:
            family.append(mask)
            fam_set.add(mask)
        
        scores = [0] * total_masks
        
        def update_scores_for_addition(mask, delta):
            # Optimized: iterate only over comparable masks
            # A mask s is comparable to mask if (s | mask) == s or (s | mask) == mask
            # i.e., s is subset of mask or mask is subset of s
            # Subsets of mask: iterate over submasks
            # Supersets of mask: iterate over supermasks
            # This is still O(2^n) in worst case but can be optimized
            
            # For now, keep the simple approach but it's O(4096) per update
            for s in range(total_masks):
                if (s | mask) == s or (s | mask) == mask:
                    scores[s] += delta
        
        for mask in full_chain:
            update_scores_for_addition(mask, 1)
        
        # Greedy initialization
        while len(family) < m:
            best_score = -1
            best_mask = -1
            for s in range(total_masks):
                if s in fam_set:
                    continue
                if scores[s] > best_score:
                    best_score = scores[s]
                    best_mask = s
            if best_mask == -1:
                break
            family.append(best_mask)
            fam_set.add(best_mask)
            update_scores_for_addition(best_mask, 1)
        
        def get_total_score():
            return (sum(scores[s] for s in family) - m) // 2
        
        current_score = get_total_score()
        best_family = family[:]
        best_score_val = current_score
        
        # Simulated Annealing
        T = 30.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() - start_time < time_budget and T > min_T:
            # 1-swap with better candidate selection
            i = random.randint(0, m - 1)
            A = family[i]
            
            # Pick B from a biased distribution: prefer masks with high scores
            # Use rejection sampling with score-based bias
            B = -1
            attempts = 0
            while B == -1 and attempts < 20:
                # Pick a random mask, accept with probability proportional to score
                B = random.randint(0, total_masks - 1)
                if B not in fam_set:
                    # Accept with probability based on score relative to max possible
                    max_possible = m  # max comparable is m-1 others + self = m
                    p = min(1.0, scores[B] / max(1, max_possible * 0.5))
                    if random.random() < p:
                        break
                B = -1
                attempts += 1
            
            if B == -1:
                # Fallback: pick any non-family mask
                while True:
                    B = random.randint(0, total_masks - 1)
                    if B not in fam_set:
                        break
            
            comp_AB = 1 if is_comparable(A, B) else 0
            gain = (scores[B] - comp_AB) - (scores[A] - 1)
            
            if gain > 0 or random.random() < math.exp(gain / T):
                fam_set.remove(A)
                fam_set.add(B)
                family[i] = B
                
                update_scores_for_addition(A, -1)
                update_scores_for_addition(B, 1)
                
                new_score = get_total_score()
                if new_score > best_score_val:
                    best_score_val = new_score
                    best_family = family[:]
            
            T *= cooling
        
        # Polish pass: greedy 1-swap until convergence
        improved = True
        polish_start = time.time()
        while improved and time.time() - polish_start < 2.0:
            improved = False
            for i in range(m):
                A = family[i]
                best_gain = 0
                best_B = -1
                for B in range(total_masks):
                    if B in fam_set:
                        continue
                    comp_AB = 1 if is_comparable(A, B) else 0
                    gain = (scores[B] - comp_AB) - (scores[A] - 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_B = B
                if best_B != -1:
                    fam_set.remove(A)
                    fam_set.add(best_B)
                    family[i] = best_B
                    update_scores_for_addition(A, -1)
                    update_scores_for_addition(best_B, 1)
                    new_score = get_total_score()
                    if new_score > best_score_val:
                        best_score_val = new_score
                        best_family = family[:]
                    improved = True
                    break
        
        return best_family, best_score_val
    
    # Multi-start: try several different seeds
    best_family_overall = None
    best_score_overall = -1
    
    num_starts = 5
    time_per_start = (deadline - time.time()) / num_starts - 1  # 1 sec safety margin
    
    for seed in range(num_starts):
        remaining = deadline - time.time()
        if remaining < 5:  # Need at least 5 sec for one more attempt
            break
        
        t_budget = min(time_per_start, remaining - 3)
        if t_budget < 2:
            break
        
        fam, score = run_single_attempt(seed * 12345 + 67890, t_budget)
        
        if score > best_score_overall:
            best_score_overall = score
            best_family_overall = fam[:]
    
    if best_family_overall is None:
        # Fallback: return the greedy result
        family = []
        fam_set = set()
        full_chain = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
        for mask in full_chain:
            family.append(mask)
            fam_set.add(mask)
        scores = [0] * total_masks
        for mask in full_chain:
            for s in range(total_masks):
                if (s | mask) == s or (s | mask) == mask:
                    scores[s] += 1
        while len(family) < m:
            best_score = -1
            best_mask = -1
            for s in range(total_masks):
                if s in fam_set:
                    continue
                if scores[s] > best_score:
                    best_score = scores[s]
                    best_mask = s
            if best_mask == -1:
                break
            family.append(best_mask)
            fam_set.add(best_mask)
            for s in range(total_masks):
                if (s | best_mask) == s or (s | best_mask) == best_mask:
                    scores[s] += 1
        best_family_overall = family
    
    return _convert(best_family_overall)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
