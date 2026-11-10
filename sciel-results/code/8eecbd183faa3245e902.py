import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety

    n = 256
    candidates = list(range(1, n + 1))

    best_size = 0
    best_A = []
    best_B = []

    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check Sidon property for new sums in S
        # New sums are x+a for a in S, and 2x
        # We need to ensure these are not in sums_S and are unique among themselves.

        # 1. Check 2x
        if 2 * x in sums_S:
            return False

        # 2. Check x+a
        for a in S:
            if x + a in sums_S:
                return False

        # Check disjoint differences with other set
        for a in S:
            d = abs(x - a)
            if d in diffs_other:
                return False

        return True

    def add_to_set(x, S, sums_S, diffs_S):
        sums_S.add(2 * x)
        for a in S:
            sums_S.add(x + a)
        for a in S:
            diffs_S.add(abs(x - a))
        S.append(x)

    def remove_from_set(x, S, sums_S, diffs_S):
        # Remove all sums involving x
        for a in S:
            if a != x:
                sums_S.discard(x + a)
        sums_S.discard(2 * x)
        # Remove all diffs involving x
        for a in S:
            if a != x:
                diffs_S.discard(abs(x - a))
        S.remove(x)

    def greedy_construct(order, prefer_A_first=True):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            if time.time() - start_time > time_limit:
                break
            if prefer_A_first:
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
            else:
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                elif can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
        return A, B

    def try_improve(A, B):
        """Try local search: remove elements to allow adding others, net gain"""
        nonlocal best_size, best_A, best_B
        A = A[:]
        B = B[:]
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Rebuild sums and diffs from scratch
        for i, a in enumerate(A):
            sums_A.add(2 * a)
            for j, b in enumerate(A):
                if i < j:
                    sums_A.add(a + b)
                    diffs_A.add(abs(a - b))
        for i, b in enumerate(B):
            sums_B.add(2 * b)
            for j, c in enumerate(B):
                if i < j:
                    sums_B.add(b + c)
                    diffs_B.add(abs(b - c))

        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            # Try removing one element from A and adding up to 2 from remaining
            for i in range(len(A)):
                if time.time() - start_time > time_limit - 5:
                    break
                x = A[i]
                # Temporarily remove x
                remove_from_set(x, A, sums_A, diffs_A)
                
                # Try to add elements back
                added = []
                for y in candidates:
                    if y == x or y in A:
                        continue
                    if can_add(y, A, sums_A, diffs_A, diffs_B):
                        add_to_set(y, A, sums_A, diffs_A)
                        added.append(y)
                        if len(added) >= 2:
                            break
                
                if len(added) > 0:
                    improved = True
                    new_size = len(A) + len(B)
                    if new_size > best_size:
                        best_size = new_size
                        best_A = A[:]
                        best_B = B[:]
                else:
                    # Put x back
                    add_to_set(x, A, sums_A, diffs_A)
            
            # Same for B
            for i in range(len(B)):
                if time.time() - start_time > time_limit - 5:
                    break
                x = B[i]
                remove_from_set(x, B, sums_B, diffs_B)
                
                added = []
                for y in candidates:
                    if y == x or y in B:
                        continue
                    if can_add(y, B, sums_B, diffs_B, diffs_A):
                        add_to_set(y, B, sums_B, diffs_B)
                        added.append(y)
                        if len(added) >= 2:
                            break
                
                if len(added) > 0:
                    improved = True
                    new_size = len(A) + len(B)
                    if new_size > best_size:
                        best_size = new_size
                        best_A = A[:]
                        best_B = B[:]
                else:
                    add_to_set(x, B, sums_B, diffs_B)
        
        return A, B

    # Strategy 1: Sorted greedy (deterministic baseline)
    A1, B1 = greedy_construct(sorted(candidates), True)
    size1 = len(A1) + len(B1)
    if size1 > best_size:
        best_size = size1
        best_A = A1[:]
        best_B = B1[:]

    # Strategy 2: Sorted greedy, prefer B first
    A2, B2 = greedy_construct(sorted(candidates), False)
    size2 = len(A2) + len(B2)
    if size2 > best_size:
        best_size = size2
        best_A = A2[:]
        best_B = B2[:]

    # Strategy 3: Reverse sorted greedy
    A3, B3 = greedy_construct(sorted(candidates, reverse=True), True)
    size3 = len(A3) + len(B3)
    if size3 > best_size:
        best_size = size3
        best_A = A3[:]
        best_B = B3[:]

    # Strategy 4: Random restarts with local search improvement
    max_restarts = 100000
    restart_count = 0
    last_improve_time = start_time

    while time.time() - start_time < time_limit and restart_count < max_restarts:
        restart_count += 1
        shuffled = candidates[:]
        random.shuffle(shuffled)

        # Alternate which set to prefer
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(shuffled, prefer_A)

        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            last_improve_time = time.time()
        
        # If we've found a good solution, do some targeted local search
        if best_size >= 20 and (restart_count % 5 == 0 or current_size >= best_size - 1):
            if time.time() - start_time < time_limit - 10:
                A_improved, B_improved = try_improve(best_A, best_B)
                improved_size = len(A_improved) + len(B_improved)
                if improved_size > best_size:
                    best_size = improved_size
                    best_A = A_improved[:]
                    best_B = B_improved[:]
                    last_improve_time = time.time()

    if best_size == 0:
        return [[1], [2]]

    return [best_A, best_B]
