import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    def is_safe_add(x, B):
        # x is 1-indexed. bit index is x-1.
        # Check if adding x creates a 4-AP.
        
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+3d <= N
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if ((B >> (x + d - 1)) & 1) and ((B >> (x + 2 * d - 1)) & 1) and ((B >> (x + 3 * d - 1)) & 1):
                return False

        # Case 2: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d >= 1
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if ((B >> (x - 3 * d - 1)) & 1) and ((B >> (x - 2 * d - 1)) & 1) and ((B >> (x - d - 1)) & 1):
                return False

        # Case 3: x is 2nd term: x-d, x, x+d, x+2d
        # Need x+2d <= N
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if ((B >> (x - d - 1)) & 1) and ((B >> (x + d - 1)) & 1) and ((B >> (x + 2 * d - 1)) & 1):
                return False

        # Case 4: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d >= 1
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if ((B >> (x - 2 * d - 1)) & 1) and ((B >> (x - d - 1)) & 1) and ((B >> (x + d - 1)) & 1):
                return False

        return True

    def greedy_randomized(B_init=0):
        B = B_init
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if not (B & (1 << (x - 1))):
                if is_safe_add(x, B):
                    B |= (1 << (x - 1))
        return B

    def local_search_add(B):
        # Add any valid element in increasing order
        # This is fast and deterministic.
        for x in range(1, N + 1):
            if not (B & (1 << (x - 1))):
                if is_safe_add(x, B):
                    B |= (1 << (x - 1))
        return B

    def local_search_swap(B, max_swaps=20):
        # Try to improve by removing one element and refilling
        best_B = B
        best_size = bin(B).count("1")
        
        for _ in range(max_swaps):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random element to remove
            bits = [i+1 for i in range(N) if (B >> i) & 1]
            if not bits:
                break
            v = random.choice(bits)
            B_temp = B & ~(1 << (v - 1))
            
            # Greedy fill on B_temp with random order
            B_temp = greedy_randomized(B_temp)
            
            size = bin(B_temp).count("1")
            if size > best_size:
                best_B = B_temp
                best_size = size
                B = best_B # Continue searching from the new best
        
        return best_B

    best_B = 0
    best_size = 0
    
    # Initial best from a few random runs
    for _ in range(10):
        B = greedy_randomized()
        B = local_search_add(B)
        B = local_search_swap(B)
        size = bin(B).count("1")
        if size > best_size:
            best_B = B
            best_size = size
            
    # Main loop
    while time.time() - start_time < time_limit:
        # Generate new candidate
        B = greedy_randomized()
        
        # Local search to maximize
        B = local_search_add(B)
        
        # Try swap improvements
        B = local_search_swap(B)
        
        size = bin(B).count("1")
        if size > best_size:
            best_B = B
            best_size = size
            
    # Convert bitset to list
    result = [i + 1 for i in range(N) if (best_B >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
