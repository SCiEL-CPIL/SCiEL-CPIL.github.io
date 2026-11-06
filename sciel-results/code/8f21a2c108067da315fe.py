import random
import itertools
import time

def is_sidon_k(S):
    """Check if a set of k integers is Sidon (all pair sums i<=j unique)."""
    k = len(S)
    sums = set()
    for i in range(k):
        for j in range(i, k):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_5(S):
    """Count the number of 5-element subsets that are Sidon."""
    count = 0
    for sub in itertools.combinations(S, 5):
        if is_sidon_k(sub):
            count += 1
    return count

def construct():
    start_time = time.time()
    best_S = None
    best_count = float('inf')
    
    # Initial random sets
    for restart in range(50):
        if time.time() - start_time > 55:
            break
            
        # Initialize with random distinct integers
        S = random.sample(range(129), 12)
        current_count = count_sidon_5(S)
        
        if current_count < best_count:
            best_count = current_count
            best_S = S[:]
            
        if current_count == 0:
            return S
            
        # Local Search
        for step in range(2000):
            if time.time() - start_time > 55:
                break
                
            if current_count == 0:
                return S
                
            # Pick a random index to change
            idx = random.randint(0, 11)
            old_val = S[idx]
            
            # Try to find a new value that reduces the count
            # We sample a few new values
            found_improvement = False
            for _ in range(50):
                new_val = random.randint(0, 128)
                if new_val == old_val or new_val in S:
                    continue
                
                S[idx] = new_val
                new_count = count_sidon_5(S)
                
                if new_count < current_count:
                    current_count = new_count
                    found_improvement = True
                    break
                else:
                    S[idx] = old_val
            
            if not found_improvement:
                # If stuck, make a random move to escape local minima
                # This is a simple random walk step
                new_val = random.randint(0, 128)
                # Ensure distinctness is handled by retrying if collision, 
                # but for the sake of speed in the inner loop, 
                # we just pick a random valid one from the remaining pool or just retry
                # Actually, simpler: pick a random index and value, ensure distinct
                # But we already picked idx. Let's just pick a new random value for idx
                # and if it collides, we'll fix it in the next step or accept the collision?
                # No, S must have distinct elements.
                # Let's pick a new value that is distinct.
                available = [v for v in range(129) if v not in S]
                if available:
                    new_val = random.choice(available)
                    S[idx] = new_val
                    current_count = count_sidon_5(S)
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_S = S[:]
                        
            # Update best
            if current_count < best_count:
                best_count = current_count
                best_S = S[:]
                
    # If we didn't find 0, return the best we found
    # The task asks to minimize the max subset size. 
    # If best_count > 0, it means there is at least one Sidon 5-subset.
    # However, the evaluator computes the *largest* Sidon subset.
    # If we have a Sidon 5-subset, the max size is at least 5.
    # If we have no Sidon 5-subset, the max size is at most 4.
    # Returning the set with the fewest Sidon 5-subsets is a good proxy, 
    # but ideally we want 0.
    # Given the constraints (12 items in 129), it is highly likely a solution with 0 exists.
    return best_S
