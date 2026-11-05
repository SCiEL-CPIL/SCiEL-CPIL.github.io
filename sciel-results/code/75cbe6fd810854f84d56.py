import time
import random

def construct():
    # Deadline guard: 90s limit, use 55s for safety
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def is_valid(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        if any(x < 1 or x > 256 for x in A) or any(x < 1 or x > 256 for x in B):
            return False
        
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                
        # Check Disjoint Non-zero Diffs
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[i] - A[j]
                    diffs_A.add(d)
                    
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[i] - B[j]
                    diffs_B.add(d)
                    
        if diffs_A & diffs_B:
            return False
            
        return True

    def try_add(x, A, sums_A, diffs_A, diffs_B):
        # Check Sidon sums for A
        new_sums_bits = 1 << (x + x)
        for a in A:
            new_sums_bits |= 1 << (x + a)
        
        if new_sums_bits & sums_A:
            return False
        
        # Check Difference constraints
        new_diffs_bits = 0
        for a in A:
            d1 = x - a
            d2 = a - x
            new_diffs_bits |= 1 << (d1 + 255)
            new_diffs_bits |= 1 << (d2 + 255)
            
        if new_diffs_bits & (diffs_A | diffs_B):
            return False
            
        return True

    def add_to_set(x, A, sums_A, diffs_A):
        sums_A |= 1 << (x + x)
        new_diffs_bits = 0
        for a in A:
            sums_A |= 1 << (x + a)
            d1 = x - a
            d2 = a - x
            new_diffs_bits |= 1 << (d1 + 255)
            new_diffs_bits |= 1 << (d2 + 255)
        diffs_A |= new_diffs_bits
        A.append(x)

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts)
        
        candidates = list(range(1, 257))
        
        strategy = attempts % 3
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            pass 
        else:
            candidates.sort(reverse=True)
            
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        for x in candidates:
            valid_A = try_add(x, A, sums_A, diffs_A, diffs_B)
            valid_B = try_add(x, B, sums_B, diffs_A, diffs_B)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    add_to_set(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, sums_A, diffs_A)
                    else:
                        add_to_set(x, B, sums_B, diffs_B)
            elif valid_A:
                add_to_set(x, A, sums_A, diffs_A)
            elif valid_B:
                add_to_set(x, B, sums_B, diffs_B)
                
        # Ensure non-empty and valid
        if not A:
            A = [1]
            sums_A = 1 << 2
            diffs_A = 0
        if not B:
            B = [2]
            sums_B = 1 << 4
            diffs_B = 0
            
        # Final Validation
        if is_valid(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A.copy()
                best_B = B.copy()
        else:
            # If invalid, try to fix by removing conflicting elements?
            # Or just skip this attempt.
            pass

    # Final check before returning
    if not is_valid(best_A, best_B):
        return [[1], [2]]
        
    return [best_A, best_B]
