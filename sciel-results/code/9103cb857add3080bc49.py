import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 256
    
    def can_add(x, S, sums_S, D_S, D_Other):
        """
        Check if x can be added to S.
        S: list of elements
        sums_S: set of all pair sums a+b (a<=b) in S
        D_S: set of all absolute differences |a-b| in S
        D_Other: set of all absolute differences in the other set
        """
        # Check Sidon property for S (sums)
        for a in S:
            s = x + a
            if s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
        
        # Check Sidon property for S (differences)
        # And Disjoint Difference constraint
        for a in S:
            d = abs(x - a)
            if d == 0: continue
            # d must not be in D_S (to keep S Sidon)
            if d in D_S:
                return False
            # d must not be in D_Other (to keep differences disjoint)
            if d in D_Other:
                return False
        
        return True

    def update_structs(x, S, sums_S, D_S):
        """
        Add x to S and update sums_S and D_S.
        Returns new S, sums_S, D_S (as sets/lists)
        Note: We should copy or modify in place carefully. 
        Since we are in a greedy loop, we can modify in place and revert if needed, 
        but here we just build forward.
        """
        new_S = S + [x]
        new_sums = set(sums_S)
        new_D = set(D_S)
        
        for a in S:
            new_sums.add(x + a)
            d = abs(x - a)
            new_D.add(d)
        new_sums.add(2 * x)
        
        return new_S, new_sums, new_D

    def remove_structs(x, S, sums_S, D_S):
        """
        Remove x from S and update sums_S and D_S.
        Returns new S, sums_S, D_S.
        """
        new_S = [a for a in S if a != x]
        new_sums = set(sums_S)
        new_D = set(D_S)
        
        # Remove sums involving x
        for a in new_S:
            new_sums.discard(x + a)
        new_sums.discard(2 * x)
        
        # Remove diffs involving x
        for a in new_S:
            d = abs(x - a)
            new_D.discard(d)
            
        return new_S, new_sums, new_D

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        D_A = set()
        D_B = set()
        
        for x in order:
            can_A = can_add(x, A, sums_A, D_A, D_B)
            can_B = can_add(x, B, sums_B, D_B, D_A)
            
            if can_A and can_B:
                # Choose randomly or balance sizes
                if len(A) <= len(B):
                    A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
                else:
                    B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
            elif can_A:
                A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
            elif can_B:
                B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
        
        return A, B, sums_A, sums_B, D_A, D_B

    def local_search(A, B, sums_A, sums_B, D_A, D_B, rng, time_limit):
        """
        Perform local search to improve the size of A and B.
        """
        used = set(A) | set(B)
        pool = [x for x in range(1, 257) if x not in used]
        
        # 1. Try to add any element from pool
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Shuffle pool to avoid bias
            rng.shuffle(pool)
            for x in pool:
                can_A = can_add(x, A, sums_A, D_A, D_B)
                can_B = can_add(x, B, sums_B, D_B, D_A)
                if can_A and can_B:
                    if len(A) <= len(B):
                        A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
                    else:
                        B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
                    pool.remove(x)
                    changed = True
                    break # Restart loop with new pool
                elif can_A:
                    A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
                    pool.remove(x)
                    changed = True
                    break
                elif can_B:
                    B, sums_B, D_B = update_structs(x, B, sums_B, D_B)
                    pool.remove(x)
                    changed = True
                    break
        
        # 2. Try swaps: Remove x from A, try to add y from pool or B to A
        # This is more expensive, so we do fewer iterations
        for _ in range(50):
            if time.time() >= time_limit:
                break
            
            # Pick a random element from A or B to remove
            if not A or not B:
                continue
                
            if rng.random() < 0.5:
                if not A: continue
                idx = rng.randint(0, len(A) - 1)
                x = A[idx]
                A, sums_A, D_A = remove_structs(x, A, sums_A, D_A)
                
                # Try to add something to A
                candidates = pool + B
                rng.shuffle(candidates)
                found = False
                for y in candidates:
                    if can_add(y, A, sums_A, D_A, D_B):
                        A, sums_A, D_A = update_structs(y, A, sums_A, D_A)
                        if y in pool:
                            pool.remove(y)
                        else:
                            B, sums_B, D_B = remove_structs(y, B, sums_B, D_B)
                        # Put x back into pool? Or try to add it to B?
                        # Let's just put x in pool for now, it might be added later
                        pool.append(x)
                        found = True
                        break
                
                if not found:
                    # Revert: put x back in A
                    A, sums_A, D_A = update_structs(x, A, sums_A, D_A)
                    # If we removed y from B, we need to put it back? No, we only removed y if we added it.
                    # If we didn't add y, we didn't remove it.
                    # Wait, the logic above: if we found y, we added it and removed it from source.
                    # If we didn't find y, we reverted x.
                    # This is correct.
            else:
                if not B: continue
                idx = rng.randint(0, len(B) - 1)
                x = B[idx]
                B, sums_B, D_B = remove_structs(x, B, sums_B, D_B)
                
                candidates = pool + A
                rng.shuffle(candidates)
                found = False
                for y in candidates:
                    if can_add(y, B, sums_B, D_B, D_A):
                        B, sums_B, D_B = update_structs(y, B, sums_B, D_B)
                        if y in pool:
                            pool.remove(y)
                        else:
                            A, sums_A, D_A = remove_structs(y, A, sums_A, D_A)
                        pool.append(x)
                        found = True
                        break
                
                if not found:
                    B, sums_B, D_B = update_structs(x, B, sums_B, D_B)

        return A, B, sums_A, sums_B, D_A, D_B

    best_A = []
    best_B = []
    best_size = 0
    
    # Initial deterministic run to have a baseline
    base_order = list(range(1, n + 1))
    A0, B0, sA0, sB0, dA0, dB0 = greedy_construct(base_order)
    A0, B0, sA0, sB0, dA0, dB0 = local_search(A0, B0, sA0, sB0, dA0, dB0, random.Random(0), start_time + 1.0)
    best_A, best_B = A0, B0
    best_size = len(A0) + len(B0)
    
    # Randomized restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random permutation
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        A, B, sA, sB, dA, dB = greedy_construct(order)
        
        # Local search with a small time budget per iteration
        # Estimate time per local search
        local_time_budget = max(0.1, (deadline - time.time()) * 0.05)
        if local_time_budget > 0.05:
            A, B, sA, sB, dA, dB = local_search(A, B, sA, sB, dA, dB, rng, time.time() + local_time_budget)
        
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B

    return [best_A, best_B]
