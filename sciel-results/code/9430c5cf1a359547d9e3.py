import time
import random

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Initial superincreasing sequence
    A = [1 << i for i in range(n)]
    best_A = A[:]
    best_max = max(A)

    def is_valid(arr):
        # Check if all subset sums are distinct
        # n=12, 4096 sums
        sums = [0]
        for x in arr:
            # Extend with x + s for all s in sums
            # Using list comprehension is fast
            sums += [s + x for s in sums]
        return len(set(sums)) == (1 << n)

    # Local Search
    # We want to minimize max(A)
    # Strategy:
    # 1. Identify the largest element.
    # 2. Try to replace it with a smaller one.
    # 3. If that fails, try to modify other elements to "make room" for a smaller max?
    #    Or just random moves on any element to escape local minima.
    
    current_A = A[:]
    
    # Helper to get a random index biased towards larger values
    def get_biased_index(arr):
        # Probability proportional to value? Or just pick max?
        # Let's pick max with prob 0.5, random otherwise.
        if random.random() < 0.5:
            return max(range(n), key=lambda i: arr[i])
        else:
            return random.randint(0, n - 1)

    while time.time() < deadline:
        i = get_biased_index(current_A)
        old_val = current_A[i]
        
        # Try to find a smaller valid value
        # Randomly sample some smaller values
        # To be efficient, sample a few
        found_better = False
        for _ in range(20): # 20 random attempts per loop
            if current_A[i] <= 1:
                break
            # Sample from [1, current_A[i] - 1]
            # Bias towards smaller values? Or uniform?
            # Uniform might be good to explore.
            v = random.randint(1, current_A[i] - 1)
            
            # Check distinctness of elements
            if v in current_A:
                # If v is at index i, it's the same value, skip
                if current_A[i] == v:
                    continue
                # If v is at another index, skip
                # We need to check if v is in current_A excluding index i?
                # No, if v is in current_A at j != i, then set is not distinct.
                # So if v in current_A, and current_A[i] != v, then v is taken.
                # If current_A[i] == v, we didn't change it.
                # So: if v in current_A and current_A[i] != v: continue
                # Actually, simpler:
                # Make a copy or check carefully.
                # Since n is small, just check:
                if v in current_A:
                    # If the only occurrence is at i, and we are changing i, it's fine?
                    # No, if we change A[i] to v, and v was at A[j], then A[j]=v and A[i]=v. Collision.
                    # So v must not be in A[j] for any j != i.
                    # So if v in current_A, and current_A[i] != v, then v is taken by someone else.
                    # If current_A[i] == v, we are not changing it effectively.
                    continue
            
            current_A[i] = v
            
            if is_valid(current_A):
                # Accept
                new_max = max(current_A)
                if new_max < best_max:
                    best_max = new_max
                    best_A = current_A[:]
                found_better = True
                break
            else:
                # Revert
                current_A[i] = old_val
        
        if not found_better:
            # If we couldn't improve by reducing, maybe we should try a random move to escape?
            # Or just continue.
            pass

    return best_A
