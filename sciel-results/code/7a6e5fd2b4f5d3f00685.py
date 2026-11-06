import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    
    def check_sidon(sums_set, candidate, current_set):
        # Check if adding candidate to current_set keeps sums unique
        # sums_set contains all valid sums for current_set
        # We need to check if any new sum collides with existing sums
        # New sums are candidate + x for x in current_set, and candidate + candidate
        for x in current_set:
            s = candidate + x
            if s in sums_set:
                return False
        s = candidate + candidate
        if s in sums_set:
            return False
        return True

    def check_diffs(dis_set, candidate, current_set):
        # Check if adding candidate to current_set keeps diffs disjoint from other set's diffs
        # dis_set is the difference set of the OTHER set
        # We need to check if any new diff from current_set + candidate intersects dis_set
        # New diffs are candidate - x and x - candidate for x in current_set
        for x in current_set:
            d1 = candidate - x
            d2 = x - candidate
            if d1 in dis_set or d2 in dis_set:
                return False
        return True

    def update_sum_set(sums_set, candidate, current_set):
        for x in current_set:
            sums_set.add(candidate + x)
        sums_set.add(candidate + candidate)

    def update_diff_set(diff_set, candidate, current_set):
        for x in current_set:
            diff_set.add(candidate - x)
            diff_set.add(x - candidate)

    def try_add(A, B, sums_A, sums_B, diffs_A, diffs_B, x, side):
        """
        Try to add x to side (0 for A, 1 for B).
        Returns True if added, False otherwise.
        Modifies sets in place if successful.
        """
        if side == 0:
            if check_sidon(sums_A, x, A) and check_diffs(diffs_B, x, A):
                # Update structures
                A.append(x)
                update_sum_set(sums_A, x, A[:-1]) # A[:-1] is current before append? No, A was appended.
                # Wait, A.append(x) happened. Let's be careful.
                # Let's revert logic: check first, then update.
                # I will rewrite this part to be safer.
                return True
        else:
            if check_sidon(sums_B, x, B) and check_diffs(diffs_A, x, B):
                return True
        return False

    # Better implementation of the greedy step to avoid state confusion
    def greedy_build():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # We alternate or randomize side selection
        # A simple strategy: for each candidate, try to add to A, then B, or random side.
        # To balance sizes, we might prefer the smaller set.
        
        for x in candidates:
            # Decide side: prefer smaller set, or random
            if len(A) < len(B):
                side = 0
            elif len(B) < len(A):
                side = 1
            else:
                side = random.choice([0, 1])
            
            # Try preferred side
            if side == 0:
                # Check A
                ok_sum = True
                for a in A:
                    if (x + a) in sums_A:
                        ok_sum = False
                        break
                if ok_sum and (x + x) not in sums_A:
                    # Check diffs against B
                    ok_diff = True
                    for a in A:
                        if (x - a) in diffs_B or (a - x) in diffs_B:
                            ok_diff = False
                            break
                    if ok_diff:
                        A.append(x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                        sums_A.add(x + x)
                        for a in A[:-1]:
                            diffs_A.add(x - a)
                            diffs_A.add(a - x)
                        continue
            else:
                # Check B
                ok_sum = True
                for b in B:
                    if (x + b) in sums_B:
                        ok_sum = False
                        break
                if ok_sum and (x + x) not in sums_B:
                    # Check diffs against A
                    ok_diff = True
                    for b in B:
                        if (x - b) in diffs_A or (b - x) in diffs_A:
                            ok_diff = False
                            break
                    if ok_diff:
                        B.append(x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                        sums_B.add(x + x)
                        for b in B[:-1]:
                            diffs_B.add(x - b)
                            diffs_B.add(b - x)
                        continue
            
            # If preferred side failed, try the other side
            other_side = 1 - side
            if other_side == 0:
                ok_sum = True
                for a in A:
                    if (x + a) in sums_A:
                        ok_sum = False
                        break
                if ok_sum and (x + x) not in sums_A:
                    ok_diff = True
                    for a in A:
                        if (x - a) in diffs_B or (a - x) in diffs_B:
                            ok_diff = False
                            break
                    if ok_diff:
                        A.append(x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                        sums_A.add(x + x)
                        for a in A[:-1]:
                            diffs_A.add(x - a)
                            diffs_A.add(a - x)
            else:
                ok_sum = True
                for b in B:
                    if (x + b) in sums_B:
                        ok_sum = False
                        break
                if ok_sum and (x + x) not in sums_B:
                    ok_diff = True
                    for b in B:
                        if (x - b) in diffs_A or (b - x) in diffs_A:
                            ok_diff = False
                            break
                    if ok_diff:
                        B.append(x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                        sums_B.add(x + x)
                        for b in B[:-1]:
                            diffs_B.add(x - b)
                            diffs_B.add(b - x)
                            
        return A, B

    # Run multiple random restarts
    while time.time() - start_time < time_limit:
        A, B = greedy_build()
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            
            # Optional: Local search improvement
            # Try to add more elements by shuffling remaining candidates?
            # The greedy above processes all candidates.
            # If we improve, we can try to refine.
            
            # If we hit a high score, we might spend a bit more time on local search
            # but for now, restarts are fast.

    return [best_A, best_B]
