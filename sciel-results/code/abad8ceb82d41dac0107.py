import itertools
import math

def construct():
    # Search for a set U of size 5 or 6 from 0..15
    # that maximizes the score
    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    # To keep it fast, we limit the search space
    # We fix 0 in the set
    # We try sizes 5 and 6
    # We try elements from 0 to 15
    
    candidates = []
    for size in [5, 6]:
        # Generate combinations of size-1 elements from 1..15
        # This is C(15, 4) = 1365 for size 5
        # C(15, 5) = 3003 for size 6
        # Total ~4000 sets. Very fast.
        for comb in itertools.combinations(range(1, 16), size - 1):
            u = [0] + list(comb)
            u.sort()
            
            # Calculate score
            sums = set()
            diffs = set()
            max_u = u[-1]
            
            for i in u:
                for j in u:
                    sums.add(i + j)
                    diffs.add(i - j)
            
            if not sums or not diffs:
                continue
                
            ratio = len(diffs) / len(sums)
            if ratio <= 1:
                continue
                
            denom = math.log2(2 * max_u + 1)
            if denom == 0:
                continue
                
            score = 1 + math.log2(ratio) / denom
            
            if score > best_score:
                best_score = score
                best_set = u
                
    return best_set
