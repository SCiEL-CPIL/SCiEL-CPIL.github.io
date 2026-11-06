# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return time_limit - elapsed()
    
    def compute_sums_valid(A):
        """Check if A is a valid Sidon set (all pair sums including a+a are unique)."""
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def compute_diffs(A):
        """Compute nonzero difference set of A."""
        diffs = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[j] - A[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(x, A):
        """Check if x can be added to Sidon set A."""
        for a in A:
            s = x + a
            # Check if s already exists as a sum in A
            for i in range(len(A)):
                for j in range(i, len(A)):
                    if A[i] + A[j] == s:
                        # The only exception is if this is x+x and x not in A, or x+a where a=x
                        # Actually we need to be more careful
                        pass
        # More efficient: build sum set and check
        sum_set = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_set.add(A[i] + A[j])
        for a in A:
            if x + a in sum_set:
                return False
        if x + x in sum_set:
            return False
        return True
    
    def can_add_diff(x, A, diffs_other):
        """Check if adding x to A creates differences that conflict with diffs_other."""
        for a in A:
            d1 = x - a
            d2 = a - x
            if d1 != 0 and d1 in diffs_other:
                return False
            if d2 != 0 and d2 in diffs_other:
                return False
        return True
    
    def balanced_greedy(order_a, order_b, max_iter=200000):
        """Balanced greedy: alternate between adding to A and B."""
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        def get_sums(S):
            s = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s.add(S[i] + S[j])
            return s
        
        def get_diffs(S):
            d = set()
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        dd = S[j] - S[i]
                        if dd != 0:
                            d.add(dd)
            return d
        
        iter_count = 0
        while time_left() > 1.0 and iter_count < max_iter:
            iter_count += 1
            progress = False
            
            # Try to add to A
            for x in order_a:
                if x in A:
                    continue
                # Check Sidon for A
                new_sums = set()
                valid = True
                for a in A:
                    s = x + a
                    if s in sums_A:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    if x + x in sums_A:
                        valid = False
                    else:
                        new_sums.add(x + x)
                if not valid:
                    continue
                # Check difference disjointness
                new_diffs = set()
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 != 0:
                        new_diffs.add(d1)
                    if d2 != 0:
                        new_diffs.add(d2)
                if new_diffs & diffs_B:
                    continue
                # Accept
                A.append(x)
                sums_A = sums_A | new_sums
                diffs_A = diffs_A | new_diffs
                progress = True
                break
            
            if time_left() < 1.0:
                break
            
            # Try to add to B
            for x in order_b:
                if x in B:
                    continue
                # Check Sidon for B
                new_sums = set()
                valid = True
                for b in B:
                    s = x + b
                    if s in sums_B:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    if x + x in sums_B:
                        valid = False
                    else:
                        new_sums.add(x + x)
                if not valid:
                    continue
                # Check difference disjointness
                new_diffs = set()
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    if d1 != 0:
                        new_diffs.add(d1)
                    if d2 != 0:
                        new_diffs.add(d2)
                if new_diffs & diffs_A:
                    continue
                # Accept
                B.append(x)
                sums_B = sums_B | new_sums
                diffs_B = diffs_B | new_diffs
                progress = True
                break
            
            if not progress:
                break
        
        return A, B
    
    def polish(A, B, time_budget=2.0):
        """Try to improve by swapping elements."""
        start_polish = time.time()
        
        def compute_sums(S):
            s = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s.add(S[i] + S[j])
            return s
        
        def compute_diffs(S):
            d = set()
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        dd = S[j] - S[i]
                        if dd != 0:
                            d.add(dd)
            return d
        
        A = list(A)
        B = list(B)
        improved = True
        while improved and (time.time() - start_polish) < time_budget and time_left() > 1.0:
            improved = False
            # Try removing an element from A and adding a different one
            for i in range(len(A)):
                old = A[i]
                A_no_old = A[:i] + A[i+1:]
                sums_no_old = compute_sums(A_no_old)
                diffs_no_old = compute_diffs(A_no_old)
                diffs_B = compute_diffs(B)
                for x in range(1, 257):
                    if x in A_no_old:
                        continue
                    # Check Sidon
                    valid = True
                    for a in A_no_old:
                        if x + a in sums_no_old:
                            valid = False
                            break
                    if valid and x + x in sums_no_old:
                        valid = False
                    if not valid:
                        continue
                    # Check diffs
                    new_diffs = set()
                    for a in A_no_old:
                        d1 = x - a
                        d2 = a - x
                        if d1 != 0:
                            new_diffs.add(d1)
                        if d2 != 0:
                            new_diffs.add(d2)
                    if new_diffs & diffs_B:
                        continue
                    # Accept swap
                    A[i] = x
                    improved = True
                    break
                if improved:
                    break
            # Similarly for B
            if time_left() < 1.0:
                break
            for i in range(len(B)):
                old = B[i]
                B_no_old = B[:i] + B[i+1:]
                sums_no_old = compute_sums(B_no_old)
                diffs_no_old = compute_diffs(B_no_old)
                diffs_A = compute_diffs(A)
                for x in range(1, 257):
                    if x in B_no_old:
                        continue
                    # Check Sidon
                    valid = True
                    for b in B_no_old:
                        if x + b in sums_no_old:
                            valid = False
                            break
                    if valid and x + x in sums_no_old:
                        valid = False
                    if not valid:
                        continue
                    # Check diffs
                    new_diffs = set()
                    for b in B_no_old:
                        d1 = x - b
                        d2 = b - x
                        if d1 != 0:
                            new_diffs.add(d1)
                        if d2 != 0:
                            new_diffs.add(d2)
                    if new_diffs & diffs_A:
                        continue
                    # Accept swap
                    B[i] = x
                    improved = True
                    break
                if improved:
                    break
        
        return A, B
    
    best_A = []
    best_B = []
    best_size = 0
    
    # Multiple random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time_left() < 3.0:
            break
        
        # Create random orderings
        order_a = list(range(1, 257))
        order_b = list(range(1, 257))
        random.shuffle(order_a)
        random.shuffle(order_b)
        
        A, B = balanced_greedy(order_a, order_b)
        
        # Polish
        A, B = polish(A, B, time_budget=min(2.0, max(0.5, time_left() - 1.0)))
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # If no valid result, use a simple fallback
    if not best_A or not best_B:
        # Simple sequential construction
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 5, 9, 17, 33, 65, 129, 256]
        # Verify
        if not compute_sums_valid(best_A):
            best_A = [1, 2, 4, 8, 16, 32]
        if not compute_sums_valid(best_B):
            best_B = [3, 5, 9, 17, 33, 65]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
