import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    N = 1000
    max_sum = 2 * N
    
    def greedy_init(order):
        """
        Greedy construction: iterate through `order` (a permutation of 1..N),
        add each element if it keeps all pair-sum counts <= 2.
        """
        counts = [0] * (max_sum + 1)
        in_set = [False] * (N + 1)
        S = []
        
        for x in order:
            # Check if x can be added
            valid = True
            for y in S:
                if counts[x + y] == 2:
                    valid = False
                    break
            if valid and counts[2 * x] == 2:
                valid = False
                
            if valid:
                # Add x
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                in_set[x] = True
                S.append(x)
                
        return S, counts, in_set

    def local_search(S, counts, in_set, time_limit):
        """
        Improve S via local search: 
        1. Try to add any available element.
        2. If no addition possible, try swaps (remove x, add y) that maintain validity 
           and potentially reduce congestion or allow future additions.
        """
        current_size = len(S)
        best_size = current_size
        best_S = list(S)
        best_counts = [c for c in counts]
        best_in_set = [b for b in in_set]
        
        # Precompute addable set? No, it changes. 
        # We'll check on the fly.
        
        # Helper to check if a swap x -> y is valid
        def is_swap_valid(x, y):
            # x is in S, y is not in S.
            # We remove x and add y.
            # Sums involving x: x+z (z in S, z!=x), 2x.
            # Sums involving y: y+z (z in S, z!=x), 2y.
            
            # The only sums that change are those involving x or y.
            # Actually, removing x decreases counts of x+z and 2x by 1.
            # Adding y increases counts of y+z and 2y by 1.
            
            # We need to ensure no count exceeds 2 after these changes.
            
            # 1. Check 2y:
            #    new_count[2y] = counts[2y] + 1
            #    (2y != 2x since x != y)
            if counts[2 * y] + 1 > 2:
                return False
            
            # 2. Check y+z for z in S, z != x:
            #    new_count[y+z] = counts[y+z] - (1 if x+z == y+z else 0) + 1
            #    Since x != y, x+z != y+z. So the term is just counts[y+z] + 1?
            #    Wait.
            #    count_{S'}(s) where s = y+z.
            #    In S, the pairs summing to s are (a,b) in SxS such that a+b=s.
            #    When we remove x, we lose pairs (x, z') where x+z'=s. Here s=y+z.
            #    So we lose (x, y) if x+y = y+z => x=z. But z is in S and x is in S.
            #    If z=x, then s=2x. But we are iterating z != x.
            #    So for z != x, does removing x affect count[y+z]?
            #    It does if x + w = y + z for some w in S.
            #    This is getting complicated to check analytically in O(1).
            
            # Let's use a simpler check:
            # The new count for sum s is:
            #   old_count[s] 
            #   - (number of pairs (a,b) in S such that a+b=s and (a==x or b==x))
            #   + (number of pairs (a,b) in S' such that a+b=s and (a==y or b==y))
            
            # For s = y+z (z in S\{x}):
            #   Pairs in S involving x that sum to y+z:
            #     x + w = y+z => w = y+z-x.
            #     If w in S and w != x, then we lose 1.
            #     If w == x, then 2x = y+z.
            #   Pairs in S' involving y that sum to y+z:
            #     y + w = y+z => w = z. Since z in S', this pair (y,z) is added. So +1.
            
            # So new_count[y+z] = counts[y+z] - (1 if (y+z-x) in in_set and (y+z-x) != x else 0) + 1
            #                    - (1 if 2x == y+z else 0)  # This case is covered if w=x? 
            #                    Actually if w=x, then y+z-x=x => y+z=2x. 
            #                    The condition "(y+z-x) in in_set" includes x. 
            #                    But pair (x,x) is counted once in counts. 
            #                    If we remove x, we lose the pair (x,x) if 2x=s.
            #                    So the subtraction is correct if we treat (x,x) as a pair.
            
            # Let's simplify by just simulating the changes on the counts array for the candidate swap.
            # It's O(|S|) per check. |S| ~ 100. 100 * 100 = 10,000 ops. Fast enough.
            
            # Simulate removal of x
            for z in S:
                if z != x:
                    counts[x + z] -= 1
            counts[2 * x] -= 1
            in_set[x] = False
            
            # Check if y can be added to current S (which no longer has x)
            valid = True
            for z in S:
                if counts[z + y] == 2:
                    valid = False
                    break
            if valid and counts[2 * y] == 2:
                valid = False
            
            # Revert removal of x
            for z in S:
                if z != x:
                    counts[x + z] += 1
            counts[2 * x] += 1
            in_set[x] = True
            
            return valid

        while time.time() < time_limit:
            # Phase 1: Try to add any element
            added = False
            # We can iterate through all non-in_set elements and check if addable
            # To speed up, we can maintain a list of "candidates" but it's complex.
            # Just iterate 1..N. 1000 checks * |S| ~ 100 = 100k ops. ~0.01s.
            # So we can do this every loop iteration.
            
            for y in range(1, N + 1):
                if in_set[y]:
                    continue
                valid = True
                for x in S:
                    if counts[x + y] == 2:
                        valid = False
                        break
                if valid and counts[2 * y] == 2:
                    valid = False
                
                if valid:
                    # Add y
                    for x in S:
                        counts[x + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                        best_counts = list(counts)
                        best_in_set = list(in_set)
                    break
            
            if added:
                continue
            
            # Phase 2: No addition possible. Try swaps.
            # We want to find a swap x->y that is valid.
            # We accept the swap if it reduces "congestion" (sum of counts^2) 
            # or simply accept it with some probability to escape local minima?
            # Given time limits, let's accept valid swaps that reduce congestion.
            # If no such swap, maybe accept random valid swaps to move around?
            
            # Calculate congestion
            current_congestion = sum(c * c for c in counts)
            
            found_swap = False
            # Try a limited number of random swaps
            num_attempts = 50
            for _ in range(num_attempts):
                # Pick random x in S
                x = random.choice(S)
                # Pick random y not in S
                # To avoid O(N) list creation, use rejection sampling
                while True:
                    y = random.randint(1, N)
                    if not in_set[y]:
                        break
                
                if is_swap_valid(x, y):
                    # Calculate new congestion
                    # We need to simulate again or derive formula.
                    # Let's simulate to be safe, but we already did it in is_swap_valid.
                    # We can modify is_swap_valid to return new congestion or just re-simulate.
                    # Re-simulating is cheap.
                    
                    # Simulate swap
                    for z in S:
                        if z != x:
                            counts[x + z] -= 1
                    counts[2 * x] -= 1
                    in_set[x] = False
                    S.remove(x)
                    
                    for z in S:
                        counts[z + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    
                    new_congestion = sum(c * c for c in counts)
                    
                    # Accept if congestion decreased
                    if new_congestion <= current_congestion:
                        current_congestion = new_congestion
                        found_swap = True
                        break
                    else:
                        # Revert
                        for z in S:
                            if z != y:
                                counts[z + y] -= 1
                        counts[2 * y] -= 1
                        in_set[y] = False
                        S.remove(y)
                        
                        for z in S:
                            counts[x + z] += 1
                        counts[2 * x] += 1
                        in_set[x] = True
                        S.append(x)
            
            if not found_swap:
                # If we can't find an improving swap, we might be stuck.
                # We can try random restarts or just continue with random valid swaps (hill climbing with noise).
                # For now, let's accept any valid swap with a small probability to allow exploration.
                # Or just break? No, we have time.
                # Let's accept a random valid swap even if congestion increases, with prob 0.1.
                if random.random() < 0.1:
                    for _ in range(10):
                        x = random.choice(S)
                        while True:
                            y = random.randint(1, N)
                            if not in_set[y]:
                                break
                        if is_swap_valid(x, y):
                            for z in S:
                                if z != x:
                                    counts[x + z] -= 1
                            counts[2 * x] -= 1
                            in_set[x] = False
                            S.remove(x)
                            
                            for z in S:
                                counts[z + y] += 1
                            counts[2 * y] += 1
                            in_set[y] = True
                            S.append(y)
                            current_congestion = sum(c * c for c in counts)
                            break
    
    # Main Logic
    best_global_size = 0
    best_global_S = []
    best_global_counts = []
    best_global_in_set = []
    
    # Run multiple independent greedy + local search iterations
    # Time budget for each iteration?
    # Total time 135s. Let's do iterations until deadline.
    
    while time.time() < deadline:
        # Greedy Init
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S, counts, in_set = greedy_init(candidates)
        
        # Local Search
        # Give local search a time budget, say 10s per run
        # Or just run it until global deadline?
        # If we run it until global deadline, we only get one run.
        # Better to have multiple shorter runs.
        # Let's say 10s per run.
        
        ls_start = time.time()
        ls_limit = ls_start + 10
        
        # Check if we should stop
        if ls_limit > deadline:
            ls_limit = deadline - 1
            
        local_search(S, counts, in_set, ls_limit)
        
        if len(S) > best_global_size:
            best_global_size = len(S)
            best_global_S = list(S)
            best_global_counts = list(counts)
            best_global_in_set = list(in_set)
            
        if time.time() > deadline - 5:
            break
            
    return best_global_S

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
