import random
import time

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    # We only care if it's >= 4 or == 3
    # Check all 4-subsets
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if is_sidon([S[i], S[j], S[k], S[l]]):
                        return 4 # Or higher, but 4 is enough to say "not 3"
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5s margin
    
    # Initial random set
    S = random.sample(range(129), 12)
    best_S = S[:]
    best_size = max_sidon_size(best_S)
    
    if best_size == 3:
        return best_S
        
    current_S = S[:]
    
    while time.time() < deadline:
        # Find a 4-subset that is Sidon
        sidon_4 = None
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    for l in range(k+1, 12):
                        if is_sidon([current_S[i], current_S[j], current_S[k], current_S[l]]):
                            sidon_4 = [i, j, k, l]
                            break
                    if sidon_4: break
                if sidon_4: break
            if sidon_4: break
            
        if not sidon_4:
            # Found a set with no 4-Sidon subsets
            return current_S
            
        # Pick a random index from the Sidon 4-subset to mutate
        idx_to_change = random.choice(sidon_4)
        
        # Generate a new value
        # Avoid existing values
        existing = set(current_S)
        new_val = random.randint(0, 128)
        attempts = 0
        while new_val in existing and attempts < 20:
            new_val = random.randint(0, 128)
            attempts += 1
            
        if new_val in existing:
            continue
            
        old_val = current_S[idx_to_change]
        current_S[idx_to_change] = new_val
        
        # Check if we improved (reduced max size)
        # If we broke the Sidon property of that specific 4-subset, 
        # the max size might drop to 3.
        # We don't need to check full max size every time, just if it's still >=4.
        # But checking max_sidon_size is cheap (495 checks).
        new_size = max_sidon_size(current_S)
        
        if new_size < best_size:
            best_size = new_size
            best_S = current_S[:]
            if best_size == 3:
                return best_S
        else:
            # Revert if not improved? Or keep random walk?
            # Hill climbing: revert if not improved.
            current_S[idx_to_change] = old_val

    return best_S
