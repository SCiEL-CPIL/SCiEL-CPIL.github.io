import itertools
import math

def construct():
    best_score = 0
    best_set = [0, 1, 2, 4]
    
    # Search geometric progressions {0, 1, b, b^2, ...}
    # and also subsets of small ranges that are "geometric-like"
    
    # 1. Pure Geometric Progressions
    for base in range(2, 20):
        for k in range(2, 10):
            u = [0]
            val = 1
            for _ in range(k - 1):
                u.append(val)
                val *= base
                if val > 1000: # Limit max U to keep denominator reasonable
                    break
            if len(u) < 2:
                continue
            
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

    # 2. Small brute force search to find non-geometric gems
    # Extend range slightly to catch things like {0,1,3,9} if they weren't pure geometric?
    # Actually {0,1,3,9} IS pure geometric base 3.
    # Let's do a small random or grid search for sets of size 4,5,6 from 0..30
    # to see if any non-geometric set beats the best geometric one.
    
    # To keep it fast, we sample.
    # But the previous brute force was fast. Let's just expand the range of the brute force.
    # C(30, 5) is 142506. Might be too slow in Python if done naively?
    # 142k * 25 ops = 3.5M ops. Should be < 1s.
    
    for size in [4, 5, 6]:
        # We already checked 5,6 up to 15. Let's check up to 30 for size 4,5,6?
        # Or just stick to geometric which seems strong.
        # Let's add a few specific candidates known to be good:
        # {0, 1, 3, 9, 27}
        # {0, 1, 2, 4, 8}
        # {0, 1, 4, 16}
        
        pass

    return best_set
