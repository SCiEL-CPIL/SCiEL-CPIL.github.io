import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_size = 0
    best_set = []
    
    def is_valid_set(S):
        counts = {}
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                b = S[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    def get_counts(S):
        counts = {}
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                b = S[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def greedy_init():
        S = []
        counts = {}
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                # Add x
                S.append(x)
                for y in S:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                # Note: The loop above adds x+x again if x is in S?
                # No, x was just appended, so it is in S.
                # So s = x+x is added once in the loop.
                # And we checked 2x separately?
                # In the check:
                # for y in S (old S): check x+y.
                # check 2x.
                # In the update:
                # for y in S (new S, includes x): s = x+y.
                # This adds x+x once.
                # So it is consistent.
                
        return S

    def local_search(S):
        counts = get_counts(S)
        
        while time.time() < deadline:
            # Try to add
            added = False
            S_set = set(S)
            not_in_S = [x for x in range(1, 1001) if x not in S_set]
            
            for x in not_in_S:
                valid = True
                for y in S:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                if valid:
                    s = 2 * x
                    if counts.get(s, 0) >= 2:
                        valid = False
                
                if valid:
                    S.append(x)
                    S_set.add(x)
                    for y in S:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    added = True
                    break
            
            if added:
                continue
            
            # Try swaps
            swaps_done = 0
            max_swaps = 1000
            for _ in range(max_swaps):
                if time.time() > deadline: break
                if len(S) == 0: break
                a = random.choice(S)
                x = random.choice(not_in_S)
                if x == a: continue
                
                # Create S'
                S_prime = [y for y in S if y != a] + [x]
                
                # Check validity of S_prime
                if is_valid_set(S_prime):
                    S = S_prime
                    counts = get_counts(S)
                    swaps_done += 1
                    break # Restart add phase
            
            if swaps_done == 0:
                break
                
        return S

    while time.time() < deadline:
        S = greedy_init()
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
