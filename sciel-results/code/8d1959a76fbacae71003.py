import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    
    best_result = []
    best_count = 0
    
    # Precompute: for each triple, what are the "completing" triples needed
    # For triple (a,b,c), a K(2,2,2) containing it needs pairs {a,a'}, {b,b'}, {c,c'}
    # and all 8 transversals. The 7 others are determined by a',b',c'.
    
    def creates_k222(triple, present_set):
        a, b, c = triple
        for a2 in range(n):
            if a2 == a:
                continue
            for b2 in range(n):
                if b2 == b:
                    continue
                for c2 in range(n):
                    if c2 == c:
                        continue
                    # Check if all 7 other transversals are present
                    # The 8 transversals of pairs {a,a2},{b,b2},{c,c2}:
                    # (a,b,c) - this is our triple, skip
                    # (a,b,c2), (a,b2,c), (a,b2,c2), (a2,b,c), (a2,b,c2), (a2,b2,c), (a2,b2,c2)
                    if (a, b, c2) not in present_set:
                        continue
                    if (a, b2, c) not in present_set:
                        continue
                    if (a, b2, c2) not in present_set:
                        continue
                    if (a2, b, c) not in present_set:
                        continue
                    if (a2, b, c2) not in present_set:
                        continue
                    if (a2, b2, c) not in present_set:
                        continue
                    if (a2, b2, c2) not in present_set:
                        continue
                    return True
        return False
    
    # Greedy with random restarts
    num_restarts = 0
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_set = set()
        current_list = []
        
        for t in all_triples:
            if not creates_k222(t, current_set):
                current_set.add(t)
                current_list.append(t)
        
        if len(current_list) > best_count:
            best_count = len(current_list)
            best_result = list(current_list)
        
        num_restarts += 1
        # Check deadline more frequently
        if time.time() > deadline:
            break
    
    # Convert to list of lists
    return [list(t) for t in best_result]
