import time
import random
import math

def _score_of(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    val = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial best
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Define a local search function
    def local_search(U, max_val=50):
        current_U = set(U)
        current_score = _score_of(list(current_U))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for x in range(max_val + 1):
                if x not in current_U:
                    new_U = current_U | {x}
                    new_score = _score_of(list(new_U))
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        improved = True
            # Try removing
            for x in list(current_U):
                if x == 0:
                    continue
                new_U = current_U - {x}
                new_score = _score_of(list(new_U))
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    improved = True
            # Try swapping
            for x in list(current_U):
                if x == 0:
                    continue
                for y in range(max_val + 1):
                    if y in current_U:
                        continue
                    new_U = (current_U - {x}) | {y}
                    new_score = _score_of(list(new_U))
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        improved = True
        return list(current_U), current_score

    # Run local search from seed
    best_U, best_score = local_search(best_U)
    
    # Random restarts
    while time.time() < deadline:
        # Random set
        size = random.randint(3, 12)
        max_el = random.randint(5, 60)
        U = set()
        U.add(0)
        while len(U) < size:
            U.add(random.randint(1, max_el))
        U = list(U)
        
        # Local search from this random start
        new_U, new_score = local_search(U, max_val=60)
        
        if new_score > best_score:
            best_U = new_U
            best_score = new_score

    return best_U
