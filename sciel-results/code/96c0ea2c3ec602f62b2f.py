import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # safety margin before 90s kill

def _pair_sums(els):
    """Return set of all a+b for a<=b in els."""
    s = set()
    for i in range(len(els)):
        for j in range(i, len(els)):
            s.add(els[i] + els[j])
    return s

def _nonzero_diffs(els):
    """Return set of all a-b for a!=b in els."""
    d = set()
    for i in range(len(els)):
        for j in range(len(els)):
            if i != j:
                d.add(els[i] - els[j])
    return d

def _can_add_x(els, x, other_diffs):
    """Check if x can be added to els maintaining Sidon and disjoint diffs."""
    new_diffs = set()
    for a in els:
        d = x - a
        if d in other_diffs or d in new_diffs:
            return False
        new_diffs.add(d)
        d2 = a - x
        if d2 in other_diffs or d2 in new_diffs:
            return False
        new_diffs.add(d2)
    # Check Sidon sums
    sums = _pair_sums(els)
    for a in els:
        s = a + x
        if s in sums:
            return False
    # x+x
    if 2*x in sums:
        return False
    return True

def _greedy_build():
    """Greedy build of two Sidon sets with disjoint differences."""
    A = []
    B = []
    diffs_A = set()
    diffs_B = set()
    
    for x in range(1, N + 1):
        # Try adding to A
        ok_A = _can_add_x(A, x, diffs_B)
        # Try adding to B
        ok_B = _can_add_x(B, x, diffs_A)
        
        if ok_A and ok_B:
            # Add to the smaller set
            if len(A) <= len(B):
                A.append(x)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            else:
                B.append(x)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        elif ok_A:
            A.append(x)
            for a in A[:-1]:
                diffs_A.add(x - a)
                diffs_A.add(a - x)
        elif ok_B:
            B.append(x)
            for b in B[:-1]:
                diffs_B.add(x - b)
                diffs_B.add(b - x)
    
    return A, B

def _local_search(A, B, time_limit=5.0):
    """Local search to improve total size."""
    start = time.time()
    A = list(A)
    B = list(B)
    best_size = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    def rebuild_diffs(els):
        d = set()
        for i in range(len(els)):
            for j in range(len(els)):
                if i != j:
                    d.add(els[i] - els[j])
        return d
    
    while time.time() - start < time_limit:
        diffs_A = rebuild_diffs(A)
        diffs_B = rebuild_diffs(B)
        
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if x in A or x in B:
                continue
            if len(A) < MAX_SIZE and _can_add_x(A, x, diffs_B):
                A.append(x)
                best_A = list(A)
                best_B = list(B)
                best_size = len(A) + len(B)
                improved = True
                break
            if len(B) < MAX_SIZE and _can_add_x(B, x, diffs_A):
                B.append(x)
                best_A = list(A)
                best_B = list(B)
                best_size = len(A) + len(B)
                improved = True
                break
        
        if not improved:
            # Try swap: remove one from A, try to add to B, etc.
            swapped = False
            for i, a in enumerate(A):
                if len(A) <= 1:
                    break
                A_temp = A[:i] + A[i+1:]
                new_diffs_A = rebuild_diffs(A_temp)
                # Try to add a to B
                if len(B) < MAX_SIZE and _can_add_x(B, a, new_diffs_A):
                    A = A_temp
                    B.append(a)
                    best_A = list(A)
                    best_B = list(B)
                    best_size = len(A) + len(B)
                    swapped = True
                    improved = True
                    break
                # Try to find a replacement for a in A
                for x in range(1, N + 1):
                    if x in A_temp or x in B:
                        continue
                    if _can_add_x(A_temp, x, diffs_B):
                        A_temp.append(x)
                        A = A_temp
                        best_A = list(A)
                        best_B = list(B)
                        best_size = len(A) + len(B)
                        swapped = True
                        improved = True
                        break
                if swapped:
                    break
            if not swapped:
                for i, b in enumerate(B):
                    if len(B) <= 1:
                        break
                    B_temp = B[:i] + B[i+1:]
                    new_diffs_B = rebuild_diffs(B_temp)
                    if len(A) < MAX_SIZE and _can_add_x(A, b, new_diffs_B):
                        B = B_temp
                        A.append(b)
                        best_A = list(A)
                        best_B = list(B)
                        best_size = len(A) + len(B)
                        swapped = True
                        improved = True
                        break
                    for x in range(1, N + 1):
                        if x in A or x in B_temp:
                            continue
                        if _can_add_x(B_temp, x, diffs_A):
                            B_temp.append(x)
                            B = B_temp
                            best_A = list(A)
                            best_B = list(B)
                            best_size = len(A) + len(B)
                            swapped = True
                            improved = True
                            break
                    if swapped:
                        break
            
            if not improved:
                # Try random restart with different ordering
                break
        
        if time.time() - start > time_limit:
            break
    
    return best_A, best_B

def construct():
    start_time = time.time()
    best_total = 0
    best_result = ([1], [2])  # fallback
    
    # Initial greedy
    A, B = _greedy_build()
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (list(A), list(B))
    
    # Local search
    A2, B2 = _local_search(A, B, time_limit=20.0)
    total2 = len(A2) + len(B2)
    if total2 > best_total:
        best_total = total2
        best_result = (list(A2), list(B2))
    
    # Random restarts with different orderings
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5.0:
        for trial in range(50):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            random.seed(trial)
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            A_r = []
            B_r = []
            diffs_A = set()
            diffs_B = set()
            
            for x in order:
                ok_A = _can_add_x(A_r, x, diffs_B)
                ok_B = _can_add_x(B_r, x, diffs_A)
                
                if ok_A and ok_B:
                    if len(A_r) <= len(B_r):
                        A_r.append(x)
                        for a in A_r[:-1]:
                            diffs_A.add(x - a)
                            diffs_A.add(a - x)
                    else:
                        B_r.append(x)
                        for b in B_r[:-1]:
                            diffs_B.add(x - b)
                            diffs_B.add(b - x)
                elif ok_A:
                    A_r.append(x)
                    for a in A_r[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                elif ok_B:
                    B_r.append(x)
                    for b in B_r[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            
            total_r = len(A_r) + len(B_r)
            if total_r > best_total:
                best_total = total_r
                best_result = (list(A_r), list(B_r))
            
            # Quick local search
            if time.time() - start_time < DEADLINE - 10.0:
                A_r2, B_r2 = _local_search(A_r, B_r, time_limit=3.0)
                total_r2 = len(A_r2) + len(B_r2)
                if total_r2 > best_total:
                    best_total = total_r2
                    best_result = (list(A_r2), list(B_r2))
    
    # Final validation
    A_final, B_final = best_result
    # Verify Sidon property
    for S in [A_final, B_final]:
        sums = _pair_sums(S)
        assert len(sums) == len(S) * (len(S) + 1) // 2, "Sidon violation"
    
    # Verify disjoint differences
    dA = _nonzero_diffs(A_final)
    dB = _nonzero_diffs(B_final)
    assert not dA & dB, "Difference overlap"
    
    return [A_final, B_final]
