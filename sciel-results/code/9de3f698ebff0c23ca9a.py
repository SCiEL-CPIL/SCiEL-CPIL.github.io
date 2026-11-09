import random
import time

def construct():
    N = 512
    # Use a safety margin to avoid timeout. The limit is 90s, so we aim for 85s.
    deadline = time.time() + 85.0
    
    def can_add(x, S):
        """
        Check if adding x to the set S (bytearray) creates a 4-AP.
        Returns True if x can be added, False otherwise.
        """
        # Case 1: x is the first term: x, x+d, x+2d, x+3d
        # We need x+3d <= N
        d = 1
        while x + 3 * d <= N:
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return False
            d += 1
            
        # Case 2: x is the second term: x-d, x, x+d, x+2d
        # We need x-d >= 1 and x+2d <= N
        d = 1
        while x - d >= 1 and x + 2 * d <= N:
            if S[x - d] and S[x + d] and S[x + 2 * d]:
                return False
            d += 1
            
        # Case 3: x is the third term: x-2d, x-d, x, x+d
        # We need x-2d >= 1 and x+d <= N
        d = 1
        while x - 2 * d >= 1 and x + d <= N:
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return False
            d += 1
            
        # Case 4: x is the fourth term: x-3d, x-2d, x-d, x
        # We need x-3d >= 1
        d = 1
        while x - 3 * d >= 1:
            if S[x - 3 * d] and S[x - 2 * d] and S[x - d]:
                return False
            d += 1
                
        return True

    def base4_seed():
        """
        Generate a 4-AP-free set using base-4 digits 0 and 1 only.
        This is a known construction for large 4-AP-free sets.
        """
        S = bytearray(N + 1)
        for num in range(1, N + 1):
            n = num
            valid = True
            while n > 0:
                digit = n % 4
                if digit > 1:
                    valid = False
                    break
                n //= 4
            if valid:
                S[num] = 1
        return S

    def greedy_complete(S_init):
        """
        Take an initial set S_init and greedily add remaining elements
        in random order if they don't form a 4-AP.
        """
        S = S_init.copy()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if not S[x]: # If x is not already in the set
                if can_add(x, S):
                    S[x] = 1
        return S

    def local_search(S, max_rounds=20):
        """
        Swap-based local search.
        Try removing one element and then greedily adding others to see if size increases.
        """
        S = S.copy()
        current_size = sum(S)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            elements = [i for i in range(1, N + 1) if S[i]]
            random.shuffle(elements)
            
            # Try a subset of elements to remove to save time
            # We limit the number of removal attempts per round
            max_removals_to_try = min(len(elements), 50)
            
            for u in elements[:max_removals_to_try]:
                if time.time() > deadline:
                    return S, current_size
                
                S_temp = S.copy()
                S_temp[u] = 0
                
                # Try to add any element not in S_temp
                # We iterate through all possible additions
                added = 0
                for v in range(1, N + 1):
                    if not S_temp[v]:
                        if can_add(v, S_temp):
                            S_temp[v] = 1
                            added += 1
                
                new_size = current_size - 1 + added
                if new_size > current_size:
                    S = S_temp
                    current_size = new_size
                    improved = True
                    break # Restart the round with the new best set
                    
        return S, current_size

    # Phase 1: Base-4 Seed
    best_arr = base4_seed()
    best_size = sum(best_arr)
    
    # Phase 2: Greedy Completion from Base-4 Seed
    # Do a few random completions from the base-4 seed
    for _ in range(5):
        if time.time() > deadline:
            break
        S = greedy_complete(best_arr)
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S
    
    # Phase 3: Randomized Greedy from Empty
    # Perform many random greedy runs from scratch to find potentially better structures
    # Since can_add is fast with bytearray, we can do more restarts
    num_greedy_restarts = 0
    while time.time() < deadline - 10.0: # Keep 10s for local search
        num_greedy_restarts += 1
        S = bytearray(N + 1)
        S = greedy_complete(S)
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = S
            
    # Phase 4: Local Search on the best set found
    if time.time() < deadline - 2.0:
        best_arr, best_size = local_search(best_arr)
        
    # Phase 5: Final Polish
    # Try to add any remaining element one last time
    for v in range(1, N + 1):
        if not best_arr[v]:
            if can_add(v, best_arr):
                best_arr[v] = 1
                best_size += 1

    # Convert bytearray to list of integers for the return interface
    result = [i for i in range(1, N + 1) if best_arr[i]]
    return result

