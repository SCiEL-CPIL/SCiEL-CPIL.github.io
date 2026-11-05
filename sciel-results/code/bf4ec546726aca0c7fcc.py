import time
import random

def construct():
    deadline = time.time() + 55.0  # safety margin before 90s kill
    best_size = 0
    best_A = []
    best_B = []
    n = 256
    max_each = 64

    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check differences against other set's diffs
        for s in S:
            d = abs(x - s)
            if d in diffs_other:
                return False
        # Check sums: x+s for s in S, and 2*x
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
        return True

    def add_to(x, S, sums_S, diffs_S):
        for s in S:
            sums_S.add(x + s)
            diffs_S.add(abs(x - s))
        sums_S.add(2 * x)
        S.append(x)

    def remove_from(x, S, sums_S, diffs_S):
        idx = S.index(x)
        S.pop(idx)
        for s in S:
            sums_S.discard(x + s)
            diffs_S.discard(abs(x - s))
        sums_S.discard(2 * x)

    def try_add_any(A, B, sums_A, sums_B, diffs_A, diffs_B, order):
        """Try to add any element from order to A or B. Return True if added."""
        for x in order:
            if x in A or x in B:
                continue
            ca = can_add(x, A, sums_A, diffs_A, diffs_B)
            cb = can_add(x, B, sums_B, diffs_B, diffs_A)
            if ca and cb:
                if len(A) < len(B):
                    add_to(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    add_to(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        add_to(x, A, sums_A, diffs_A)
                    else:
                        add_to(x, B, sums_B, diffs_B)
                return True
            elif ca:
                add_to(x, A, sums_A, diffs_A)
                return True
            elif cb:
                add_to(x, B, sums_B, diffs_B)
                return True
        return False

    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Local search: try removals and re-additions, and single-element swaps."""
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try removing each element and seeing if we can add more
            for set_idx in range(2):
                S = A if set_idx == 0 else B
                sums_S = sums_A if set_idx == 0 else sums_B
                diffs_S = diffs_A if set_idx == 0 else diffs_B
                diffs_other = diffs_B if set_idx == 0 else diffs_A
                other_S = B if set_idx == 0 else A
                other_sums = sums_B if set_idx == 0 else sums_A
                other_diffs = diffs_A if set_idx == 0 else diffs_B

                # Try removing each element
                for i in range(len(S)):
                    if time.time() > deadline - 1.0:
                        break
                    x = S[i]
                    old_len = len(S)
                    remove_from(x, S, sums_S, diffs_S)
                    
                    # Try to add back x or something else
                    # First try to add x back (no-op check)
                    # Then try to add other elements
                    added = False
                    for y in range(1, n + 1):
                        if y in S or y in other_S:
                            continue
                        if can_add(y, S, sums_S, diffs_S, diffs_other):
                            add_to(y, S, sums_S, diffs_S)
                            added = True
                            break
                    if not added:
                        # Try adding to the other set
                        if can_add(x, other_S, other_sums, other_diffs, diffs_S):
                            add_to(x, other_S, other_sums, other_diffs)
                            added = True
                        else:
                            # Try adding x back to original set
                            if can_add(x, S, sums_S, diffs_S, diffs_other):
                                add_to(x, S, sums_S, diffs_S)
                            else:
                                # x can't be added anywhere, leave it out
                                pass
                    # If we added something different, improved
                    if len(S) > old_len or (added and len(S) == old_len and x not in S):
                        improved = True
                        break
                if improved:
                    break
            if time.time() > deadline - 1.0:
                break
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    # Multiple randomized restarts
    while time.time() < deadline - 2.0:
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Randomize candidate order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Greedy construction
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            ca = can_add(x, A, sums_A, diffs_A, diffs_B)
            cb = can_add(x, B, sums_B, diffs_B, diffs_A)
            
            if ca and cb:
                if len(A) < len(B):
                    add_to(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    add_to(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        add_to(x, A, sums_A, diffs_A)
                    else:
                        add_to(x, B, sums_B, diffs_B)
            elif ca:
                add_to(x, A, sums_A, diffs_A)
            elif cb:
                add_to(x, B, sums_B, diffs_B)
        
        # Local search
        A, B, sums_A, sums_B, diffs_A, diffs_B = local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        # Try more additions after local search
        order = list(range(1, n + 1))
        random.shuffle(order)
        try_add_any(A, B, sums_A, sums_B, diffs_A, diffs_B, order)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
            
            # If we've hit the theoretical max, stop early
            if best_size >= 128:
                break

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
