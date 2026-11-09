import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 256
    max_size = 64
    
    def can_add_to_A(x, A, sums_A, diffs_B):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if x - a in diffs_B:
                return False
            if a - x in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if x - b in diffs_A:
                return False
            if b - x in diffs_A:
                return False
        return True

    def get_state(A, B):
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
                    
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
                    
        return sums_A, sums_B, diffs_A, diffs_B

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= max_size and len(B) >= max_size:
                break
                
            # Try A
            if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                sums_A.add(x + x)
            # Try B
            elif len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                sums_B.add(x + x)
                
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        all_elements = set(range(1, n + 1))
        
        # We will try to improve the solution.
        # Priority: 1. Add new elements. 2. Swap/Move to allow adding.
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try to add any unused element
            unused = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline - 1:
                    break
                    
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(x + x)
                    in_A.add(x)
                    improved = True
                    break # Restart loop with new state
                    
                # Try add to B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(x + x)
                    in_B.add(x)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
                
            # 2. If no direct addition, try swaps or moves to free up space
            # Try Swaps
            # To make this efficient, we don't test all pairs every time.
            # We pick a random subset or just iterate if time permits.
            
            # Let's try a limited number of random swaps/moves
            attempts = 0
            while attempts < 20 and time.time() < deadline - 1:
                attempts += 1
                
                # Random Swap
                if random.random() < 0.5 and len(A) > 0 and len(B) > 0:
                    a_idx = random.randint(0, len(A) - 1)
                    b_idx = random.randint(0, len(B) - 1)
                    a_val = A[a_idx]
                    b_val = B[b_idx]
                    
                    # Remove a from A, b from B
                    A_temp = A[:a_idx] + A[a_idx+1:]
                    B_temp = B[:b_idx] + B[b_idx+1:]
                    
                    # Recalculate sums/diffs for temp sets? 
                    # That's O(N^2). Too slow for many swaps.
                    # We need incremental removal logic or just accept the cost 
                    # if N is small (64). 64^2 = 4096. 20 swaps = 80k ops. 
                    # In Python, this might be slow. 
                    # Let's optimize: Only do swaps if we are stuck and have time.
                    
                    # For now, let's stick to a simpler local search: 
                    # Just try to add elements. If stuck, do a few random restarts.
                    # The greedy + polish is usually enough for this size.
                    
                    # Let's implement a "k-opt" style: remove k elements, try to re-add.
                    # Too complex for the time limit.
                    
                    break # Skip complex swaps for now, rely on multiple restarts
                    
                # Random Move (A to B or B to A)
                else:
                    if len(A) > 0 and len(B) < max_size:
                        a_idx = random.randint(0, len(A) - 1)
                        a_val = A[a_idx]
                        A_temp = A[:a_idx] + A[a_idx+1:]
                        
                        # Check if a_val can go to B
                        # We need diffs of A_temp and sums of B.
                        # Recalculating is expensive. 
                        # Given the time constraint and Python speed, 
                        # multiple fast restarts are better than one slow deep search.
                        
                        break

        return A, B, sums_A, sums_B, diffs_A, diffs_B

    best_A, best_B = [], []
    best_score = 0
    
    num_trials = 200 # High number, but each trial is fast
    
    for trial in range(num_trials):
        if time.time() > deadline - 2:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B, sums_A, sums_B, diffs_A, diffs_B = greedy_construct(order)
        
        # Polish
        A, B, sums_A, sums_B, diffs_A, diffs_B = local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
            
        # Early exit if we hit theoretical max? 
        # Max Sidon size in [1, 256] is approx sqrt(256)=16? No, Sidon sets can be larger.
        # Actually, a Sidon set in [1, N] has size at most sqrt(N) + O(N^1/4).
        # For N=256, sqrt(256)=16. But we have two sets.
        # Wait, the max size of a Sidon set in [1, n] is roughly sqrt(n).
        # If max size is ~16, then |A|+|B| <= 32?
        # Let's check the seed score. 1.2941.
        # If seed is 1.0, and score is value/seed? No, "search score is value/seed".
        # The task says "Maximize |A|+|B|".
        # If the max possible is small, we should find it quickly.
        
        # Actually, known bounds: Max size of Sidon set in [1, n] is ~ sqrt(n).
        # For n=256, sqrt(256)=16.
        # So |A| <= 16, |B| <= 16. Total <= 32.
        # Let's see if we can reach 32.
        
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
