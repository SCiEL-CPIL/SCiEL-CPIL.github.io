import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    best_size = 0
    best_A = []
    best_B = []
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def have_disjoint_diffs(A, B):
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = abs(A[i] - A[j])
                if d == 0: continue
                diffs_A.add(d)
        
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                d = abs(B[i] - B[j])
                if d == 0: continue
                if d in diffs_A:
                    return False
        return True

    def is_valid(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        if not is_sidon(A):
            return False
        if not is_sidon(B):
            return False
        if not have_disjoint_diffs(A, B):
            return False
        return True

    def greedy_construct():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            can_add_A = True
            # Check Sidon sums for A
            if (x + x) in sums_A:
                can_add_A = False
            else:
                for a in A:
                    if (x + a) in sums_A:
                        can_add_A = False
                        break
            
            # Check Diff disjointness for A (new diffs vs diffs_B)
            if can_add_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        can_add_A = False
                        break
            
            if can_add_A:
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                continue
            
            # Try adding to B
            can_add_B = True
            # Check Sidon sums for B
            if (x + x) in sums_B:
                can_add_B = False
            else:
                for b in B:
                    if (x + b) in sums_B:
                        can_add_B = False
                        break
            
            # Check Diff disjointness for B (new diffs vs diffs_A)
            if can_add_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_A:
                        can_add_B = False
                        break
            
            if can_add_B:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                continue
        
        return A, B

    while time.time() < deadline:
        A, B = greedy_construct()
        
        if is_valid(A, B):
            current_size = len(A) + len(B)
            if current_size > best_size:
                best_size = current_size
                best_A = A[:]
                best_B = B[:]
                
        # Simple local search: try to swap elements
        if best_size > 0 and time.time() < deadline - 10:
            # Try to improve by swapping one element in A with one outside A
            # This is complex, so we rely on restarts primarily.
            pass

    # Ensure non-empty and valid return
    if not best_A or not best_B:
        # Fallback to a known valid small solution
        # A = [1], B = [2]
        best_A = [1]
        best_B = [2]
    elif not is_valid(best_A, best_B):
        # If best is invalid (should not happen), fallback
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
