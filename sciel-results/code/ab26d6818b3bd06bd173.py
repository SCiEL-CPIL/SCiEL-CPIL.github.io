import random
import time

def construct():
    start_time = time.time()
    deadline = 90.0
    safety_margin = 5.0
    max_time = start_time + deadline - safety_margin
    
    N = 512
    MAX_SIZE = 48
    
    def is_valid(S):
        if len(S) < 2:
            return True
        S_set = set(S)
        for a in S:
            R = S_set - {a}
            target = 2 * a
            if target > 24576:
                continue
            
            bits = 1
            for x in R:
                bits |= bits << x
            
            if (bits >> target) & 1:
                if target in R:
                    bits2 = 1
                    for x in R:
                        if x == target:
                            continue
                        bits2 |= bits2 << x
                    if (bits2 >> target) & 1:
                        return False
                else:
                    return False
        return True

    # Initial set: Greedy
    S = []
    for x in range(1, N + 1):
        if len(S) >= MAX_SIZE:
            break
        if is_valid(S + [x]):
            S.append(x)
            
    best_S = S[:]
    best_size = len(S)
    
    current_S = S[:]
    current_size = len(current_S)
    
    while time.time() < max_time:
        if current_size > best_size:
            best_S = current_S[:]
            best_size = current_size
            if best_size == MAX_SIZE:
                break
                
        # Try to add a random element
        if current_size < MAX_SIZE:
            # Pick a few random candidates
            added = False
            for _ in range(5):
                x = random.randint(1, N)
                if x not in current_S:
                    new_S = current_S + [x]
                    if is_valid(new_S):
                        current_S = new_S
                        current_size += 1
                        added = True
                        break
            if not added:
                # If we couldn't add, try to swap
                if current_size > 0:
                    x = random.choice(current_S)
                    for _ in range(5):
                        y = random.randint(1, N)
                        if y not in current_S:
                            new_S = [z if z != x else y for z in current_S]
                            if is_valid(new_S):
                                current_S = new_S
                                break
        else:
            # Already max size, but maybe we can improve "quality" for future swaps?
            # Not needed if we just want cardinality.
            pass

    return best_S
