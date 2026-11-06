import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N
    DEADLINE = time.time() + 85
    
    masks_by_weight = [[] for _ in range(MIN_DIST)]
    for k in range(MIN_DIST):
        for comb in combinations(range(N), k):
            mask = 0
            for bit in comb:
                mask |= (1 << bit)
            masks_by_weight[k].append(mask)
            
    best_code = []
    
    while time.time() < DEADLINE:
        valid_set = set(range(NUM_WORDS))
        code = []
        
        while valid_set:
            if time.time() > DEADLINE - 5:
                break
            
            # Pick random word
            # random.choice on a set is not directly available, convert to list
            # This is O(N)
            candidates = list(valid_set)
            w = random.choice(candidates)
            
            code.append(w)
            valid_set.remove(w)
            
            # Remove neighbors
            for k in range(MIN_DIST):
                for mask in masks_by_weight[k]:
                    v = w ^ mask
                    if v in valid_set:
                        valid_set.remove(v)
                        
        if len(code) > len(best_code):
            best_code = code

    return best_code
