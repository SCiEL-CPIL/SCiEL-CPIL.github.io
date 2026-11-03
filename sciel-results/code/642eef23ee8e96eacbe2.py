import math
import random
import time

def _score(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs)
    den = len(sums)
    if den == 0:
        return 0
    return 1 + math.log(num / den) / math.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 145
    
    # Known good candidate
    best_set = [0, 3, 4]
    best_score = _score(best_set)
    
    # Local search
    current = best_set[:]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding an element
        for x in range(0, 50):
            if x in current:
                continue
            new_set = current + [x]
            new_score = _score(new_set)
            if new_score > current_score:
                current = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
        # Try removing an element (except 0)
        for i in range(len(current)):
            if current[i] == 0:
                continue
            new_set = current[:i] + current[i+1:]
            new_score = _score(new_set)
            if new_score > current_score:
                current = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
        # Try replacing an element
        for i in range(len(current)):
            if current[i] == 0:
                continue
            old_val = current[i]
            for x in range(0, 50):
                if x == old_val or x in current:
                    continue
                new_set = current[:]
                new_set[i] = x
                new_score = _score(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                        
    return best_set
