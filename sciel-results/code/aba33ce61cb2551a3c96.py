import time
import random

def construct():
    deadline = time.time() + 50.0
    best_size = 0
    best_A = []
    best_B = []
    
    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check differences
        for s in S:
            d = abs(x - s)
            if d in diffs_other:
                return False
        
        # Check sums
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
            
        return True

    def update_sums_diffs(x, S, sums_S, diffs_S):
        for s in S:
            sums_S.add(x + s)
            diffs_S.add(abs(x - s))
        sums_S.add(2 * x)
        S.append(x)

    def remove_from_set(x, S, sums_S, diffs_S):
        """Remove x from set S and update sums and diffs."""
        S.remove(x)
        sums_S.discard(2 * x)
        for s in S:
            sums_S.discard(x + s)
            diffs_S.discard(abs(x - s))

    def try_local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to improve by adding elements, swapping, or restructuring."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add any element not yet in either set
            for x in range(1, 257):
                if x in A or x in B:
                    continue
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    update_sums_diffs(x, A, sums_A, diffs_A)
                    improved = True
                    break
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    update_sums_diffs(x, B, sums_B, diffs_B)
                    improved = True
                    break
            
            # Try swaps: remove an element from one set, try to add to the other
            if not improved and time.time() < deadline - 2:
                # Try removing from A and adding to B
                for a in A[:]:
                    remove_from_set(a, A, sums_A, diffs_A)
                    
                    if can_add(a, B, sums_B, diffs_B, diffs_A):
                        update_sums_diffs(a, B, sums_B, diffs_B)
                        improved = True
                    else:
                        for x in range(1, 257):
                            if x in A or x in B:
                                continue
                            if can_add(x, A, sums_A, diffs_A, diffs_B):
                                update_sums_diffs(x, A, sums_A, diffs_A)
                                improved = True
                                break
                        if not improved:
                            if can_add(a, A, sums_A, diffs_A, diffs_B):
                                update_sums_diffs(a, A, sums_A, diffs_A)
                            else:
                                A.append(a)
                                for s in A:
                                    if s != a:
                                        sums_A.add(a + s)
                                        diffs_A.add(abs(a - s))
                                sums_A.add(2 * a)
                    break
                
                # Try removing from B and adding to A
                if not improved and time.time() < deadline - 2:
                    for b in B[:]:
                        remove_from_set(b, B, sums_B, diffs_B)
                        
                        if can_add(b, A, sums_A, diffs_A, diffs_B):
                            update_sums_diffs(b, A, sums_A, diffs_A)
                            improved = True
                        else:
                            for x in range(1, 257):
                                if x in A or x in B:
                                    continue
                                if can_add(x, B, sums_B, diffs_B, diffs_A):
                                    update_sums_diffs(x, B, sums_B, diffs_B)
                                    improved = True
                                    break
                            if not improved:
                                if can_add(b, B, sums_B, diffs_B, diffs_A):
                                    update_sums_diffs(b, B, sums_B, diffs_B)
                                else:
                                    B.append(b)
                                    for s in B:
                                        if s != b:
                                            sums_B.add(b + s)
                                            diffs_B.add(abs(b - s))
                                    sums_B.add(2 * b)
                        break
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    # Run multiple random trials
    while time.time() < deadline:
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Randomize the order of candidates to explore different configurations
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            can_add_A = can_add(x, A, sums_A, diffs_A, diffs_B)
            can_add_B = can_add(x, B, sums_B, diffs_B, diffs_A)
            
            if can_add_A and can_add_B:
                # Balance: prefer the set with fewer elements
                if len(A) < len(B):
                    update_sums_diffs(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    update_sums_diffs(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        update_sums_diffs(x, A, sums_A, diffs_A)
                    else:
                        update_sums_diffs(x, B, sums_B, diffs_B)
            elif can_add_A:
                update_sums_diffs(x, A, sums_A, diffs_A)
            elif can_add_B:
                update_sums_diffs(x, B, sums_B, diffs_B)
            # else: skip x
        
        # Local search: try to add more elements
        A, B, sums_A, sums_B, diffs_A, diffs_B = try_local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
