import random
import itertools
import time

def is_sidon_4(a, b, c, d):
    # Check unique sums for i<=j
    # Sums: 2a, a+b, a+c, a+d, 2b, b+c, b+d, 2c, c+d, 2d
    sums = [
        2*a, a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    ]
    # Using a set is fast enough
    return len(set(sums)) == 10

def count_sidon_4(S):
    count = 0
    for subset in itertools.combinations(S, 4):
        if is_sidon_4(*subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    best_S = list(range(12))
    best_count = count_sidon_4(best_S)
    
    if best_count == 0:
        return best_S

    # Local search to minimize count of 4-Sidon subsets
    # Target: 0
    
    for restart in range(500):
        if time.time() - start_time > 55:
            break
            
        # Random initialization
        S = [random.randint(0, 128) for _ in range(12)]
        while len(set(S)) < 12:
            idx = random.randint(0, 11)
            S[idx] = random.randint(0, 128)
            
        current_count = count_sidon_4(S)
        
        if current_count < best_count:
            best_count = current_count
            best_S = S[:]
            
        if best_count == 0:
            return best_S

        # Hill Climbing
        for step in range(2000):
            if time.time() - start_time > 55:
                break
                
            if current_count == 0:
                return S
                
            # Pick a random 4-Sidon subset to break
            # To save time, we can just pick a random index and a random new value
            # and see if it reduces the count.
            # But targeting a known Sidon subset is smarter.
            
            # Find one 4-Sidon subset
            sidon_subsets = []
            for sub in itertools.combinations(S, 4):
                if is_sidon_4(*sub):
                    sidon_subsets.append(sub)
                    if len(sidon_subsets) >= 5: # Just grab a few
                        break
            
            if not sidon_subsets:
                return S # Should not happen if current_count > 0
                
            target_subset = random.choice(sidon_subsets)
            
            # Try to move one element of target_subset
            indices = [i for i in range(12) if S[i] in target_subset]
            random.shuffle(indices)
            
            moved = False
            for idx in indices:
                old_val = S[idx]
                # Try several new values
                for _ in range(20):
                    new_val = random.randint(0, 128)
                    if new_val == old_val or new_val in S:
                        continue
                    
                    S[idx] = new_val
                    new_count = count_sidon_4(S)
                    
                    if new_count < current_count:
                        current_count = new_count
                        moved = True
                        break
                    else:
                        S[idx] = old_val
                        
                if moved:
                    break
            
            if not moved:
                # Stuck, randomize a bit
                idx = random.randint(0, 11)
                S[idx] = random.randint(0, 128)
                while len(set(S)) < 12:
                    S[idx] = random.randint(0, 128)
                current_count = count_sidon_4(S)
                
                if current_count < best_count:
                    best_count = current_count
                    best_S = S[:]
                    if best_count == 0:
                        return best_S

    return best_S
