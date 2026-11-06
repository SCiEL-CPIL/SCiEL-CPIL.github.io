import random
import time

def construct():
    start_time = time.time()
    timeout = 100.0
    
    best_diameter = float('inf')
    best_set = None
    
    def is_valid_sidon(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
        
    def get_missing(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs.add(A[j] - A[i])
        return [d for d in range(1, 31) if d not in diffs]

    # Strategy 1: Greedy construction of a Sidon set, then optimize
    # We want to cover 1..30.
    # Let's try to build a set of size 9.
    
    for attempt in range(200): # 200 restarts
        if time.time() - start_time > timeout:
            break
            
        # Greedy initialization
        A = [0]
        max_val_limit = 100 # Start with a small bound
        used_diffs = set()
        
        for _ in range(8):
            # Try to add the smallest possible number
            found = False
            for candidate in range(1, max_val_limit + 1):
                if candidate in A:
                    continue
                # Check if adding candidate creates duplicate diff
                valid = True
                new_diffs = []
                for a in A:
                    d = abs(candidate - a)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    A.append(candidate)
                    A.sort()
                    for d in new_diffs:
                        used_diffs.add(d)
                    found = True
                    break
            
            if not found:
                # If no small number works, pick a random larger one
                # This might break the "small diameter" goal initially, but we optimize later
                while True:
                    candidate = random.randint(1, max_val_limit + 50)
                    if candidate in A:
                        continue
                    valid = True
                    new_diffs = []
                    for a in A:
                        d = abs(candidate - a)
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        A.append(candidate)
                        A.sort()
                        for d in new_diffs:
                            used_diffs.add(d)
                        break
        
        # Now A is a valid Sidon set of size 9.
        # Check if it covers 1..30
        missing = get_missing(A)
        
        if not missing:
            current_max = max(A)
            if current_max < best_diameter:
                best_diameter = current_max
                best_set = A[:]
                # Try to compress
                # Simple local search to reduce max
                T = 10.0
                for step in range(500):
                    if time.time() - start_time > timeout:
                        break
                    if not is_valid_sidon(A):
                        break
                    missing = get_missing(A)
                    if missing:
                        break
                    
                    idx = random.randint(1, len(A)-1)
                    old_val = A[idx]
                    # Try to move it down
                    new_val = random.randint(1, old_val - 1)
                    if new_val in A:
                        continue
                    # Check Sidon
                    A[idx] = new_val
                    A.sort()
                    if is_valid_sidon(A):
                        # Check missing
                        if get_missing(A):
                            A[idx] = old_val
                            A.sort()
                        else:
                            # Accept
                            pass
                    else:
                        A[idx] = old_val
                        A.sort()
                    
                    if max(A) < best_diameter:
                        best_diameter = max(A)
                        best_set = A[:]

        # If not valid (missing diffs), do local search to fix
        else:
            # Local search to cover missing diffs
            T = 10.0
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                if not is_valid_sidon(A):
                    # Repair Sidon violation
                    # Find a duplicate diff and move one element
                    # Simplified: just random restart if broken
                    break
                
                missing = get_missing(A)
                if not missing:
                    current_max = max(A)
                    if current_max < best_diameter:
                        best_diameter = current_max
                        best_set = A[:]
                    # Continue to compress
                    idx = random.randint(1, len(A)-1)
                    old_val = A[idx]
                    new_val = random.randint(1, max(1, old_val - 5))
                    if new_val not in A:
                        A[idx] = new_val
                        A.sort()
                        if is_valid_sidon(A) and not get_missing(A):
                            pass
                        else:
                            A[idx] = old_val
                            A.sort()
                    continue
                
                # Pick a missing diff
                target_d = random.choice(missing)
                # Try to create it by moving an element
                idx = random.randint(1, len(A)-1)
                old_val = A[idx]
                
                # Candidate new values that create target_d with some a in A
                candidates = set()
                for a in A:
                    if a + target_d not in A:
                        candidates.add(a + target_d)
                    if a - target_d > 0 and a - target_d not in A:
                        candidates.add(a - target_d)
                
                if candidates:
                    new_val = random.choice(list(candidates))
                    if new_val not in A:
                        A[idx] = new_val
                        A.sort()
                        # Check Sidon
                        if is_valid_sidon(A):
                            # Accept if it reduces missing count or keeps it same?
                            # We just accept and let the loop check
                            pass
                        else:
                            A[idx] = old_val
                            A.sort()
                else:
                    # Random move
                    low = max(1, old_val - 5)
                    high = min(max(A) + 5, 200)
                    if low > high:
                        continue
                    new_val = random.randint(low, high)
                    if new_val not in A:
                        A[idx] = new_val
                        A.sort()
                        if not is_valid_sidon(A):
                            A[idx] = old_val
                            A.sort()

    if best_set:
        return best_set
    else:
        # Fallback: A known valid Sidon set covering 1..30
        # We need to find one. 
        # Let's try [0, 1, 4, 10, 12, 17, 21, 23, 26, 29] again? No, it was invalid.
        # Let's return a set that is definitely valid, even if diameter is large.
        # Use the greedy result from the last iteration?
        # Or just return a PDS of size 9 in Z73 mapped to integers.
        # PDS 9 in Z73: {0, 1, 3, 7, 12, 21, 26, 34, 41} ? 
        # Let's verify this set.
        # Diffs:
        # 1, 3, 7, 12, 21, 26, 34, 41
        # 3-1=2, 7-1=6, 12-1=11, 21-1=20, 26-1=25, 34-1=33, 41-1=40
        # 7-3=4, 12-3=9, 21-3=18, 26-3=23, 34-3=31, 41-3=38
        # 12-7=5, 21-7=14, 26-7=19, 34-7=27, 41-7=34 (Dup with 34-0? No 34 is in set. 34-0=34. 41-7=34. Duplicate!)
        # So that set is not Sidon.
        
        # I will return the best_set found in the loop, or if none, a large valid set.
        # To be safe, I'll construct a valid set using the greedy method one more time with a large bound.
        A = [0]
        used_diffs = set()
        for _ in range(31): # Size 32 max
            found = False
            for candidate in range(1, 10000):
                if candidate in A:
                    continue
                valid = True
                for a in A:
                    d = candidate - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    A.append(candidate)
                    A.sort()
                    for a in A[:-1]:
                        used_diffs.add(candidate - a)
                    found = True
                    break
            if not found:
                break
        return A
