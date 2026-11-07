import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Generate all masks
    all_masks = list(range(total_masks))
    
    # Precompute popcount for sorting initial family
    def popcount(x):
        return bin(x).count('1')
        
    # Initial Family: 40 masks with smallest popcount, then smallest value
    # This creates a "ball" around the empty set, which is a strong candidate for high comparability
    sorted_masks = sorted(all_masks, key=lambda x: (popcount(x), x))
    family = sorted_masks[:m]
    
    # Helper: Check if a is comparable to b (one is strict subset of other)
    # a < b if (a & b) == a and a != b
    # a > b if (a & b) == b and a != b
    # Comparable if (a & b) == a or (a & b) == b, and a != b
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    # Calculate score of a family
    def get_score(fam):
        score = 0
        k = len(fam)
        for i in range(k):
            for j in range(i + 1, k):
                if is_comparable(fam[i], fam[j]):
                    score += 1
        return score

    current_score = get_score(family)
    
    # Precompute a set for fast lookup of membership
    family_set = set(family)
    
    # List of masks NOT in family for efficient iteration? 
    # There are 4096 - 40 = 4056 candidates.
    # Iterating over all 4096 and checking membership is O(1) per check.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through each member of the family
        for i in range(m):
            if time.time() > deadline:
                break
            A = family[i]
            
            # Calculate how many other sets in family are comparable to A
            score_A = 0
            for j in range(m):
                if i == j:
                    continue
                if is_comparable(A, family[j]):
                    score_A += 1
            
            # Try to replace A with some B not in family
            best_B = None
            best_score_B = score_A
            
            # Iterate through all possible candidates B
            for B in all_masks:
                if B in family_set:
                    continue
                # Calculate how many sets in family (excluding A) are comparable to B
                # Note: We must exclude A from the count for B, because A is being removed.
                # However, if B is comparable to A, that pair is lost.
                # The delta is:
                # New pairs involving B: count of C in F\{A} such that B~C
                # Lost pairs involving A: count of C in F\{A} such that A~C
                # So we just need count_B_excl_A.
                
                score_B = 0
                for j in range(m):
                    if i == j:
                        continue
                    if is_comparable(B, family[j]):
                        score_B += 1
                
                if score_B > best_score_B:
                    best_score_B = score_B
                    best_B = B
            
            if best_B is not None:
                # Perform swap
                family[i] = best_B
                family_set.remove(A)
                family_set.add(best_B)
                improved = True
                # We could update current_score incrementally, but it's not strictly needed for the logic,
                # just for potential debugging or stopping criteria if we wanted.
                # current_score += best_score_B - score_A
                
    # Convert masks to lists of integers
    result = []
    for mask in family:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
