import random
import time

def construct():
    start_time = time.time()
    best_set = []
    n = 4096
    
    def get_sums_involving(A, x):
        """Returns set of sums in A that involve x"""
        sums = set()
        # x is in A
        idx = A.index(x)
        # Triples (x, x, x)
        sums.add(3 * x)
        # Triples (x, x, y) where y in A, y != x
        for y in A:
            if y == x: continue
            sums.add(2 * x + y)
        # Triples (x, y, z) where y < z in A, y,z != x
        # We need to be careful with indices to avoid duplicates
        # Simpler: iterate all pairs (i, j) with i <= j
        k = len(A)
        for i in range(k):
            for j in range(i, k):
                a = A[i]
                b = A[j]
                # If x is not a or b, this triple doesn't involve x
                if a == x or b == x:
                    # Need to find the third element
                    # This logic is messy.
                    pass
        
        # Better approach:
        # Generate all sums of A, and all sums of A\{x}.
        # Sums involving x = Sums(A) - Sums(A\{x})
        # But we don't want to rebuild Sums(A).
        
        # Let's just generate sums involving x directly.
        # Types:
        # 1. 3x
        # 2. 2x + y, y in A, y != x
        # 3. x + y + z, y < z in A, y,z != x
        
        sums = set()
        sums.add(3 * x)
        for y in A:
            if y != x:
                sums.add(2 * x + y)
        
        # For x + y + z
        # y, z from A \ {x}
        A_minus_x = [a for a in A if a != x]
        k = len(A_minus_x)
        for i in range(k):
            y = A_minus_x[i]
            for j in range(i, k):
                z = A_minus_x[j]
                sums.add(x + y + z)
                
        return sums

    def can_add(A, x, sum_set):
        new_sums = set()
        new_sums.add(3 * x)
        for y in A:
            new_sums.add(2 * x + y)
        k = len(A)
        for i in range(k):
            y = A[i]
            for j in range(i, k):
                z = A[j]
                new_sums.add(x + y + z)
                
        if len(new_sums) != (k + 1) * (k + 2) // 2:
            return False, None
        if new_sums & sum_set:
            return False, None
        return True, new_sums

    def greedy_init():
        A = []
        sum_set = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            ok, new_sums = can_add(A, x, sum_set)
            if ok:
                A.append(x)
                sum_set.update(new_sums)
        return A, sum_set

    def local_search(A, sum_set):
        A = list(A)
        sum_set = set(sum_set)
        A_set = set(A)
        
        while True:
            # 1. Try to add
            added = False
            for x in range(1, n + 1):
                if x in A_set:
                    continue
                ok, new_sums = can_add(A, x, sum_set)
                if ok:
                    A.append(x)
                    A_set.add(x)
                    sum_set.update(new_sums)
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try swaps
            swapped = False
            # Iterate x not in A, y in A
            for x in range(1, n + 1):
                if x in A_set:
                    continue
                for y in A:
                    if x == y:
                        continue
                    
                    # Check swap y -> x
                    # Sums removed: S_y
                    # Sums added: S_x (in context of A\{y})
                    
                    S_y = get_sums_involving(A, y)
                    
                    # Generate S_x for A' = (A \ {y}) U {x}
                    # This is sums involving x in A'
                    S_x = set()
                    S_x.add(3 * x)
                    for a in A:
                        if a == y: continue
                        S_x.add(2 * x + a)
                    
                    A_minus_y = [a for a in A if a != y]
                    k = len(A_minus_y)
                    for i in range(k):
                        a = A_minus_y[i]
                        for j in range(i, k):
                            b = A_minus_y[j]
                            S_x.add(x + a + b)
                    
                    # Check internal collision in S_x
                    # Expected size of S_x:
                    # 1 (3x) + (K-1) (2x+a) + (K-1)(K)/2 (x+a+b)
                    # K = len(A)
                    # Size = 1 + K - 1 + (K-1)K/2 = K + (K^2 - K)/2 = (K^2 + K)/2
                    expected_size = (len(A)**2 + len(A)) // 2
                    if len(S_x) != expected_size:
                        continue
                        
                    # Check collision with remaining sums
                    # Remaining sums = sum_set - S_y
                    # If S_x intersects (sum_set - S_y), invalid
                    # Equivalent: S_x & sum_set must be subset of S_y
                    # i.e., (S_x & sum_set) <= S_y
                    # Or: (S_x & sum_set) - S_y is empty
                    
                    if (S_x & sum_set) - S_y:
                        continue
                        
                    # Valid swap
                    # Update A and sum_set
                    A.remove(y)
                    A_set.remove(y)
                    A.append(x)
                    A_set.add(x)
                    
                    # Update sum_set
                    sum_set.difference_update(S_y)
                    sum_set.update(S_x)
                    
                    swapped = True
                    break
                if swapped:
                    break
                    
            if not swapped:
                break
                
        return A

    for trial in range(50):
        if time.time() - start_time > 55:
            break
        A, sum_set = greedy_init()
        A = local_search(A, sum_set)
        if len(A) > len(best_set):
            best_set = A

    return best_set
