import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    N = 192
    max_k = 48
    
    # Helper: Get set of triple sums
    def get_triple_sums(S):
        return set(sum(t) for t in combinations(S, 3))
    
    # Helper: Check if y can be added to S
    # S is a list, y is int
    # Returns True if S U {y} is valid
    def can_add(S, y):
        if y in S:
            return False
        
        # 1. Check if y divides any triple sum in S
        # Pre-compute triple sums of S outside if possible? 
        # For now, compute inside.
        triple_sums = get_triple_sums(S)
        for t in triple_sums:
            if t % y == 0:
                return False
        
        # 2. Check if any x in S divides y+a+b for distinct a,b in S\{x,y}
        S_set = set(S)
        for x in S:
            # We need to check pairs a,b from S \ {x, y}
            # Note: if len(S) < 2, no pairs.
            others = [v for v in S if v != x and v != y]
            for a, b in combinations(others, 2):
                if (y + a + b) % x == 0:
                    return False
        return True

    # Helper: Check if S is valid
    def is_valid(S):
        if len(S) < 4:
            return True
        S_set = set(S)
        for x in S:
            others = [v for v in S if v != x]
            # Check if x divides sum of any 3 distinct others
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True

    # Greedy initialization
    def greedy_init(direction='desc'):
        S = []
        candidates = list(range(1, N+1))
        if direction == 'desc':
            candidates.sort(reverse=True)
        else:
            candidates.sort()
        
        for y in candidates:
            if time.time() > deadline:
                break
            if can_add(S, y):
                S.append(y)
                if len(S) >= max_k:
                    break
        return S

    best_set = greedy_init('desc')
    
    # Local Search
    while time.time() < deadline:
        current_set = best_set[:]
        improved = False
        
        # Try additions
        for y in range(1, N+1):
            if time.time() > deadline:
                break
            if y not in current_set:
                if can_add(current_set, y):
                    current_set.append(y)
                    improved = True
                    if len(current_set) >= max_k:
                        break
        
        if len(current_set) > len(best_set):
            best_set = current_set
        
        # Try swaps if size didn't increase or to find better config
        if len(current_set) < max_k:
            # Randomize order of x to escape local minima
            xs = current_set[:]
            random.shuffle(xs)
            for x in xs:
                if time.time() > deadline:
                    break
                for y in range(1, N+1):
                    if time.time() > deadline:
                        break
                    if y not in current_set and y != x:
                        # Check if S - {x} + {y} is valid
                        # This is equivalent to can_add(S - {x}, y)
                        # But can_add expects a list.
                        S_minus_x = [v for v in current_set if v != x]
                        if can_add(S_minus_x, y):
                            # Perform swap
                            current_set = S_minus_x + [y]
                            improved = True
                            break # Restart loop for this x? Or continue?
                            # Let's break inner y loop and restart x loop?
                            # Or just accept and continue.
        
        if not improved:
            # Perturb: Remove a random element and restart greedy from remainder
            if current_set:
                remove_idx = random.randint(0, len(current_set)-1)
                perturbed = current_set[:]
                perturbed.pop(remove_idx)
                # Greedy add to perturbed
                candidates = [v for v in range(1, N+1) if v not in perturbed]
                random.shuffle(candidates)
                for y in candidates:
                    if time.time() > deadline:
                        break
                    if can_add(perturbed, y):
                        perturbed.append(y)
                        if len(perturbed) >= max_k:
                            break
                if len(perturbed) > len(best_set):
                    best_set = perturbed
                elif len(perturbed) == len(best_set) and random.random() < 0.1:
                    # Sometimes accept equal size to escape
                    best_set = perturbed

    return best_set
