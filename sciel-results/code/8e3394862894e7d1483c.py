import random
import time
from fractions import Fraction

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 50.0  # Leave 10s margin
    
    # 1. Generate initial greedy set
    def get_greedy_set():
        selected = []
        selected_set = set()
        for x in range(1, 257):
            valid = True
            for a in selected:
                for b in selected:
                    if a == b: continue
                    # Check if x forms AP with a, b
                    # Cases:
                    # 1. a, b, x -> 2b - a == x
                    # 2. a, x, b -> 2x - a == b => 2x == a+b
                    # 3. x, a, b -> 2a - x == b => 2a == x+b
                    # 4. b, a, x -> 2a - b == x
                    # 5. b, x, a -> 2x - b == a
                    # 6. x, b, a -> 2b - x == a
                    
                    # Simpler: check if any pair in selected forms AP with x
                    # For each y in selected:
                    #   Is there z in selected such that {x,y,z} is AP?
                    #   z1 = 2y - x
                    #   z2 = 2x - y
                    #   z3 = (x+y)//2 if (x+y)%2==0
                    pass
            # The seed logic was:
            # if all(2*b-a!=x for a in selected for b in selected if a<b):
            # This is O(N^2).
            # Let's use a faster check for the initial greedy.
            valid = True
            for y in selected:
                # Check if x is part of any AP with y and some z in selected
                # z = 2y - x
                if (2*y - x) in selected_set:
                    valid = False
                    break
                # z = 2x - y
                if (2*x - y) in selected_set:
                    valid = False
                    break
                # z = (x+y)/2
                if (x+y) % 2 == 0:
                    z = (x+y)//2
                    if z in selected_set:
                        valid = False
                        break
            if valid:
                selected.append(x)
                selected_set.add(x)
        return selected, selected_set

    S, S_set = get_greedy_set()
    
    def is_valid_addition(x, current_set):
        """Check if adding x to current_set keeps it 3-AP free."""
        for y in current_set:
            # z = 2y - x
            if (2*y - x) in current_set:
                return False
            # z = 2x - y
            if (2*x - y) in current_set:
                return False
            # z = (x+y)/2
            if (x+y) % 2 == 0:
                z = (x+y)//2
                if z in current_set:
                    return False
        return True

    def get_score(S_list):
        # Using float for speed, but task wants exact?
        # "score is its floating serialization with exact numerator/denominator in details"
        # The evaluator measures the raw value (higher is better).
        # I will return the list. The evaluator computes the score.
        # For internal comparison, float is sufficient.
        return sum(1.0 / x for x in S_list)

    current_score = get_score(S)
    best_S = S[:]
    best_score = current_score
    
    # Precompute reciprocal weights
    weights = {x: 1.0 / x for x in range(1, 257)}
    
    # Local Search / Simulated Annealing
    T = 0.01
    min_T = 0.0001
    
    # List of elements not in S
    candidates = [x for x in range(1, 257) if x not in S_set]
    
    while time.time() - start_time < deadline:
        # Try a swap
        if S:
            x = random.choice(S)
            if candidates:
                y = random.choice(candidates)
                
                # Check if swapping x with y is valid
                # New set = S - {x} + {y}
                # We need to check if adding y to S-{x} is valid.
                # S-{x} is valid.
                # Check is_valid_addition(y, S_set - {x})
                # Optimization: 
                # y conflicts with z in S-{x} if {y,z,w} is AP with w in S-{x} or w=y (no).
                
                # Let's just construct the new set temporarily or check carefully.
                # Check if y forms AP with any pair in S-{x}
                valid = True
                for z in S:
                    if z == x: continue
                    # Check if y, z, w form AP where w is in S-{x} or w=y? No, w must be in S-{x} or w=y is not possible as distinct.
                    # Actually, if {y, z, w} is AP, w must be in the set.
                    # w1 = 2z - y
                    # w2 = 2y - z
                    # w3 = (y+z)//2
                    # We need to check if any of these w is in S-{x} OR if w == x?
                    # If w == x, then {y, z, x} is AP. Since x is removed, this AP is broken.
                    # So we only care if w is in S-{x}.
                    
                    w1 = 2*z - y
                    if w1 in S_set and w1 != x:
                        valid = False
                        break
                    w2 = 2*y - z
                    if w2 in S_set and w2 != x:
                        valid = False
                        break
                    if (y+z) % 2 == 0:
                        w3 = (y+z)//2
                        if w3 in S_set and w3 != x:
                            valid = False
                            break
                if valid:
                    new_score = current_score - weights[x] + weights[y]
                    delta = new_score - current_score
                    if new_score > best_score:
                        best_S = S[:]
                        best_S.remove(x)
                        best_S.append(y)
                        best_score = new_score
                    
                    # Accept move
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        S.remove(x)
                        S_set.remove(x)
                        S.append(y)
                        S_set.add(y)
                        current_score = new_score
                        candidates.remove(y)
                        candidates.append(x)
                        # Sort candidates? No, random choice is fine.
        else:
            # If S is empty (unlikely), add 1
            S.append(1)
            S_set.add(1)
            candidates.remove(1)
            current_score = 1.0
            best_S = S[:]
            best_score = 1.0

        # Try an add (if possible)
        if candidates:
            # Pick a random candidate
            y = random.choice(candidates)
            if is_valid_addition(y, S_set):
                new_score = current_score + weights[y]
                # Always better
                S.append(y)
                S_set.add(y)
                current_score = new_score
                candidates.remove(y)
                if new_score > best_score:
                    best_S = S[:]
                    best_score = new_score

        # Decay temperature
        T = max(min_T, T * 0.99)

    return best_S
