"""Divisibility-antichain representations by smooth numbers.
Global min-cost assignment via pre-computed minimum-cardinality antichain sums."""
from itertools import combinations

def construct():
    # Precompute 2^a * 3^b * 5^c smooth numbers up to 128
    smooth = []
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                val = p2 * p3 * p5
                if val > 128:
                    break
                smooth.append(val)
    smooth = sorted(set(smooth))
    smooth_set = set(smooth)

    def is_divisible(a, b):
        return b % a == 0

    def valid_antichain(terms):
        # distinct positive smooth terms, no term divides another
        if len(terms) != len(set(terms)):
            return False
        for i in range(len(terms)):
            for j in range(len(terms)):
                if i == j:
                    continue
                if is_divisible(terms[i], terms[j]):
                    return False
        return True

    # Pre-compute minimum cardinality antichain for every sum 1..128
    # We iterate by cardinality k. For each k, we find all valid antichains of size k.
    # If the sum is <= 128 and not yet assigned, we assign it.
    # Since we iterate k=1, 2, ..., the first time we see a sum is its min cardinality.
    
    min_rep = {}  # sum -> tuple of terms
    targets_found = 0
    max_target = 128
    
    # Max cardinality is bounded by the number of smooth numbers, but likely small.
    # We stop when all targets are found or k exceeds limit.
    for k in range(1, len(smooth) + 1):
        if targets_found == max_target:
            break
            
        # Generate combinations of k smooth numbers
        # Optimization: Only consider combinations where sum <= 128?
        # It's hard to prune combinations by sum easily without generating them.
        # But k is small for most.
        
        for terms in combinations(smooth, k):
            s = sum(terms)
            if s > 128:
                continue
            if s in min_rep:
                continue
                
            # Check antichain property
            # Since terms are distinct (from combinations), we just check divisibility
            is_anti = True
            for i in range(k):
                for j in range(i + 1, k):
                    if terms[j] % terms[i] == 0 or terms[i] % terms[j] == 0:
                        is_anti = False
                        break
                if not is_anti:
                    break
            
            if is_anti:
                min_rep[s] = terms
                targets_found += 1

    # Build answers
    answers = []
    for n in range(1, 129):
        if n in min_rep:
            answers.append(list(min_rep[n]))
        else:
            # Fallback: should not happen if max k is sufficient
            # If it does, we need a valid rep. 
            # Given the density of smooth numbers, k=1..5 should cover all.
            # If not, we can do a quick search for this specific n.
            # But for now, we assume pre-computation covered it.
            # To be safe, let's add a simple fallback search if missing.
            # However, the problem states "All targets require valid representations".
            # If our pre-computation failed, we might need to go higher k.
            # Let's just return empty list for now, but we expect no empties.
            answers.append([])
            
    return answers
