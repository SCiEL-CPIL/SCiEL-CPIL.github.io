"""Direct construction of two Sidon sets with disjoint difference sets."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    n = 256
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
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
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = B[j] - B[i]
                diffs_B.add(d)
                
        if diffs_A & diffs_B:
            return False
            
        return True

    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(1000): # Max restarts, but will break on time
        if elapsed() > deadline - 5.0:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Shuffle candidate elements
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
                
            # Determine priority: try to add to the smaller set first
            # If sizes are equal, pick randomly
            if len(A) < len(B):
                order = [0, 1]
            elif len(B) < len(A):
                order = [1, 0]
            else:
                order = random.sample([0, 1], 2)
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= 64:
                        continue
                    # Check Sidon for A
                    valid_sidon = True
                    for a in A:
                        if (x + a) in sums_A:
                            valid_sidon = False
                            break
                    if valid_sidon and (2*x) not in sums_A:
                        # Check Disjoint Diffs
                        valid_diffs = True
                        for a in A:
                            d = abs(x - a)
                            if d != 0 and d in diffs_B:
                                valid_diffs = False
                                break
                        if valid_diffs:
                            # Add to A
                            A.append(x)
                            for a in A[:-1]:
                                sums_A.add(x + a)
                                d = abs(x - a)
                                if d != 0:
                                    diffs_A.add(d)
                            sums_A.add(2*x)
                            added = True
                            break
                else:
                    if len(B) >= 64:
                        continue
                    # Check Sidon for B
                    valid_sidon = True
                    for b in B:
                        if (x + b) in sums_B:
                            valid_sidon = False
                            break
                    if valid_sidon and (2*x) not in sums_B:
                        # Check Disjoint Diffs
                        valid_diffs = True
                        for b in B:
                            d = abs(x - b)
                            if d != 0 and d in diffs_A:
                                valid_diffs = False
                                break
                        if valid_diffs:
                            # Add to B
                            B.append(x)
                            for b in B[:-1]:
                                sums_B.add(x + b)
                                d = abs(x - b)
                                if d != 0:
                                    diffs_B.add(d)
                            sums_B.add(2*x)
                            added = True
                            break
            
        if A and B:
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
