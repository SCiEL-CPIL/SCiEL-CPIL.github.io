import random
import time

def construct():
    n = 256
    deadline = time.time() + 55.0
    
    best_sum = 0
    best_A = []
    best_B = []
    
    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs
    
    def can_add_to_sidon(x, S, sums_set):
        """Check if x can be added to S while maintaining Sidon property.
        Returns (True, updated_sums) or (False, None)."""
        for a in S:
            s = a + x
            if s in sums_set:
                return False, None
        s = x + x
        if s in sums_set:
            return False, None
        new_sums = set(sums_set)
        for a in S:
            new_sums.add(a + x)
        new_sums.add(x + x)
        return True, new_sums
    
    def can_add_with_diff(x, S, sums_set, D_S, D_other):
        """Check if x can be added to S maintaining Sidon and difference disjointness.
        Returns (True, updated_sums, updated_diffs) or (False, None, None)."""
        ok, new_sums = can_add_to_sidon(x, S, sums_set)
        if not ok:
            return False, None, None
        new_diffs = set(D_S)
        for a in S:
            new_diffs.add(x - a)
            new_diffs.add(a - x)
        if not new_diffs.isdisjoint(D_other):
            return False, None, None
        return True, new_sums, new_diffs
    
    def try_insert(x, S, sums_set, D_S, D_other):
        """Try to insert x into S. Returns (success, new_S, new_sums, new_D_S)."""
        ok, new_sums, new_diffs = can_add_with_diff(x, S, sums_set, D_S, D_other)
        if ok:
            return True, S + [x], new_sums, new_diffs
        return False, None, None, None
    
    def local_search(A, B, max_time=5.0):
        """Try to grow A and B by inserting elements, and try swaps."""
        sums_A = set()
        for a in A:
            for b in A:
                sums_A.add(a + b)
        sums_B = set()
        for a in B:
            for b in B:
                sums_B.add(a + b)
        D_A = get_diffs(A)
        D_B = get_diffs(B)
        
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < max_time:
            improved = False
            
            # Try to insert into A
            for x in range(1, n+1):
                if x in A or x in B:
                    continue
                ok, new_A, new_sums_A, new_D_A = try_insert(x, A, sums_A, D_A, D_B)
                if ok:
                    A = new_A
                    sums_A = new_sums_A
                    D_A = new_D_A
                    improved = True
                    break
            
            # Try to insert into B
            for x in range(1, n+1):
                if x in A or x in B:
                    continue
                ok, new_B, new_sums_B, new_D_B = try_insert(x, B, sums_B, D_B, D_A)
                if ok:
                    B = new_B
                    sums_B = new_sums_B
                    D_B = new_D_B
                    improved = True
                    break
            
            # Try swap: remove a from A, try to add b to A
            # Or more useful: try to replace an element to free up differences
            # Simple swap: for a in A, for b not in A, try swapping a->b in A
            if len(A) > 1:
                for i, a in enumerate(list(A)):
                    # Try removing a and adding some x
                    A_temp = A[:i] + A[i+1:]
                    sums_A_temp = set()
                    for x in A_temp:
                        for y in A_temp:
                            sums_A_temp.add(x + y)
                    D_A_temp = get_diffs(A_temp)
                    for x in range(1, n+1):
                        if x in A_temp:
                            continue
                        ok, new_A, new_sums_A, new_D_A = try_insert(x, A_temp, sums_A_temp, D_A_temp, D_B)
                        if ok and len(new_A) == len(A):
                            # Same size but might enable future insertions
                            # Only accept if it helps (we'll check later)
                            pass
                        elif ok and len(new_A) > len(A):
                            A = new_A
                            sums_A = new_sums_A
                            D_A = new_D_A
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            
            # Try swap for B
            if len(B) > 1:
                for i, b in enumerate(list(B)):
                    B_temp = B[:i] + B[i+1:]
                    sums_B_temp = set()
                    for x in B_temp:
                        for y in B_temp:
                            sums_B_temp.add(x + y)
                    D_B_temp = get_diffs(B_temp)
                    for x in range(1, n+1):
                        if x in B_temp:
                            continue
                        ok, new_B, new_sums_B, new_D_B = try_insert(x, B_temp, sums_B_temp, D_B_temp, D_A)
                        if ok and len(new_B) > len(B):
                            B = new_B
                            sums_B = new_sums_B
                            D_B = new_D_B
                            improved = True
                            break
                    if improved:
                        break
        
        return A, B
    
    def build_interleaved(candidates, target=16):
        """Build A and B interleaved."""
        A, B = [], []
        sums_A, sums_B = set(), set()
        D_A, D_B = set(), set()
        
        for x in candidates:
            if len(A) >= target and len(B) >= target:
                break
            added = False
            # Try A first
            if len(A) < target:
                ok, ns, nd = can_add_with_diff(x, A, sums_A, D_A, D_B)
                if ok:
                    A.append(x)
                    sums_A, D_A = ns, nd
                    added = True
            # Try B
            if not added and len(B) < target:
                ok, ns, nd = can_add_with_diff(x, B, sums_B, D_B, D_A)
                if ok:
                    B.append(x)
                    sums_B, D_B = ns, nd
        return A, B
    
    def build_sequential(candidates, target=16, first_is_A=True):
        """Build one set, then the other."""
        if first_is_A:
            first, second = [], []
            sums_first, sums_second = set(), set()
            D_first, D_second = set(), set()
        else:
            second, first = [], []
            sums_second, sums_first = set(), set()
            D_second, D_first = set(), set()
        
        for x in candidates:
            if len(first) >= target:
                break
            ok, ns, nd = can_add_with_diff(x, first, sums_first, D_first, D_second)
            if ok:
                first.append(x)
                sums_first, D_first = ns, nd
        
        for x in candidates:
            if len(second) >= target:
                break
            ok, ns, nd = can_add_with_diff(x, second, sums_second, D_second, D_first)
            if ok:
                second.append(x)
                sums_second, D_second = ns, nd
        
        if first_is_A:
            return first, second
        else:
            return second, first
    
    for restart in range(3000):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        # Cycle through strategies
        strategy = restart % 4
        if strategy == 0:
            A, B = build_interleaved(candidates)
        elif strategy == 1:
            A, B = build_sequential(candidates, first_is_A=True)
        elif strategy == 2:
            A, B = build_sequential(candidates, first_is_A=False)
        else:
            # Interleaved but try B first
            A, B = build_interleaved(candidates)
            # Swap roles in next iteration implicitly
        
        # Apply local search polish
        A, B = local_search(A, B, max_time=2.0)
        
        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]
        
        if best_sum >= 32:
            break
    
    return [best_A, best_B]
