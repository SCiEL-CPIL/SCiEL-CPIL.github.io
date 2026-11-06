import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    
    # Helper to check if adding x to set A creates a 4-AP
    # x can be 1st, 2nd, 3rd, or 4th term
    def creates_4ap(x, A):
        # Case 1: x is 4th term: a, a+d, a+2d, x => a = x - 3d
        # x-3d, x-2d, x-d must be in A
        for d in range(1, (x // 3) + 1):
            if (x - 3*d) in A and (x - 2*d) in A and (x - d) in A:
                return True
        # Case 2: x is 3rd term: a, a+d, x, x+d => a = x - 2d
        # x-2d, x-d, x+d must be in A
        for d in range(1, (x // 2) + 1):
            if (x - 2*d) in A and (x - d) in A and (x + d) in A:
                return True
        # Case 3: x is 2nd term: a, x, x+d, x+2d => a = x - d
        # x-d, x+d, x+2d must be in A
        for d in range(1, (n - x) // 2 + 1):
            if (x - d) in A and (x + d) in A and (x + 2*d) in A:
                return True
        # Case 4: x is 1st term: x, x+d, x+2d, x+3d
        # x+d, x+2d, x+3d must be in A
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in A and (x + 2*d) in A and (x + 3*d) in A:
                return True
        return False

    # Calculate number of 3-APs in a set A
    def count_3aps(A):
        count = 0
        # For each pair a, c in A with a < c, check if (a+c)/2 is in A
        # To avoid double counting, we can iterate b (middle) and d
        for b in A:
            for d in range(1, n + 1):
                if (b - d) in A and (b + d) in A:
                    count += 1
        return count

    # Incremental update for 3-AP count when adding x
    def delta_3aps_add(x, A):
        delta = 0
        # x is middle: (x-d, x, x+d)
        for d in range(1, n + 1):
            if (x - d) in A and (x + d) in A:
                delta += 1
        # x is first: (x, x+d, x+2d)
        for d in range(1, (n - x) // 2 + 1):
            if (x + d) in A and (x + 2*d) in A:
                delta += 1
        # x is last: (x-2d, x-d, x)
        for d in range(1, x // 2 + 1):
            if (x - 2*d) in A and (x - d) in A:
                delta += 1
        return delta

    # Incremental update for 3-AP count when removing x
    def delta_3aps_remove(x, A):
        delta = 0
        # x is middle: (x-d, x, x+d)
        for d in range(1, n + 1):
            if (x - d) in A and (x + d) in A:
                delta -= 1
        # x is first: (x, x+d, x+2d)
        for d in range(1, (n - x) // 2 + 1):
            if (x + d) in A and (x + 2*d) in A:
                delta -= 1
        # x is last: (x-2d, x-d, x)
        for d in range(1, x // 2 + 1):
            if (x - 2*d) in A and (x - d) in A:
                delta -= 1
        return delta

    # Initialize with greedy solution
    A = set()
    for x in range(1, n + 1):
        if not creates_4ap(x, A):
            A.add(x)
    
    current_score = count_3aps(A)
    best_score = current_score
    best_A = set(A)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    
    while time.time() < deadline:
        if random.random() < 0.5:
            # Try to add an element
            candidates = [i for i in range(1, n + 1) if i not in A]
            if candidates:
                x = random.choice(candidates)
                if not creates_4ap(x, A):
                    new_score = current_score + delta_3aps_add(x, A)
                    if new_score > current_score or (T > 0.1 and random.random() < math.exp((new_score - current_score) / T)):
                        A.add(x)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_A = set(A)
                T *= cooling
        else:
            # Try to remove an element
            if A:
                x = random.choice(list(A))
                # Removing always keeps validity
                new_score = current_score + delta_3aps_remove(x, A)
                # We only want to remove if it helps (or rarely to explore, but removing usually hurts 3-AP count)
                # However, removing might allow adding a better element later.
                # Let's accept removals that don't decrease score too much, or just rely on swaps?
                # Actually, simple removal is rarely beneficial for maximizing 3-APs unless it unblocks a huge gain.
                # Let's stick to swaps or just add/remove where add is the primary driver.
                # But sometimes you need to remove to add.
                if new_score >= current_score or (T > 0.1 and random.random() < math.exp((new_score - current_score) / T)):
                    A.remove(x)
                    current_score = new_score
                    # Note: best_score should only update if we are in a better state, 
                    # but removing reduces score, so best_score won't update here unless current_score was already best and we removed? No, current_score drops.
                    # So best_A remains unchanged.
                T *= cooling

    return sorted(best_A)
