"""Divisibility-antichain representations by smooth numbers.
Pre-computes minimum-cardinality antichain representations for all targets 1..128."""
from itertools import combinations

def construct():
    # 1. Generate smooth numbers
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
    
    # 2. Precompute valid antichains for small cardinalities
    # We store the smallest cardinality found for each sum
    min_count = {}
    rep = {}
    
    # Helper to check antichain property
    def is_antichain(terms):
        # terms is a sorted tuple
        for i in range(len(terms)):
            ti = terms[i]
            for j in range(i + 1, len(terms)):
                if terms[j] % ti == 0:
                    return False
        return True

    max_target = 128
    assigned = set()
    
    # Iterate through possible cardinalities
    # We only need to go up to a small number, e.g., 6
    # Since we want minimum cardinality, we can stop early for each n
    for k in range(1, 10):
        if len(assigned) == max_target:
            break
            
        # Generate all combinations of smooth numbers of size k
        for terms in combinations(smooth, k):
            s = sum(terms)
            if s > max_target:
                continue
            if s in assigned:
                continue
                
            if is_antichain(terms):
                assigned.add(s)
                min_count[s] = k
                rep[s] = terms

    # Construct the result
    answers = []
    for n in range(1, 129):
        if n in rep:
            answers.append(list(rep[n]))
        else:
            # Fallback for any missing targets (should not happen)
            # Try a simple greedy or DFS for specific n if needed
            # For now, return empty or raise
            answers.append([])
            
    return answers
