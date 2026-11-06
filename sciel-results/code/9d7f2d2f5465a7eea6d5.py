import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0
    n = 256
    max_size = 64

    def compute_sums(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        return sums

    def compute_diffs(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs.add(A[i] - A[j])
        return diffs

    def is_valid(A, B):
        if len(A) > max_size or len(B) > max_size:
            return False
        sums_A = compute_sums(A)
        if len(sums_A) != len(A)*(len(A)+1)//2:
            return False
        sums_B = compute_sums(B)
        if len(sums_B) != len(B)*(len(B)+1)//2:
            return False
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    def can_add_to_A(x, A, sums_A, diffs_B):
        if len(A) >= max_size or x in A:
            return False
        for a in A:
            if (x+a) in sums_A:
                return False
        if (2*x) in sums_A:
            return False
        for a in A:
            if (x-a) in diffs_B or (a-x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if len(B) >= max_size or x in B:
            return False
        for b in B:
            if (x+b) in sums_B:
                return False
        if (2*x) in sums_B:
            return False
        for b in B:
            if (x-b) in diffs_A or (b-x) in diffs_A:
                return False
        return True

    def run_greedy_balanced(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            # Balanced: prefer the smaller set when both can accept
            can_A = can_add_to_A(x, A, sums_A, diffs_B)
            can_B = can_add_to_B(x, B, sums_B, diffs_A)
            
            if can_A and can_B:
                if len(A) <= len(B):
                    for a in A:
                        sums_A.add(x+a)
                        diffs_A.add(x-a)
                        diffs_A.add(a-x)
                    sums_A.add(2*x)
                    A.append(x)
                else:
                    for b in B:
                        sums_B.add(x+b)
                        diffs_B.add(x-b)
                        diffs_B.add(b-x)
                    sums_B.add(2*x)
                    B.append(x)
            elif can_A:
                for a in A:
                    sums_A.add(x+a)
                    diffs_A.add(x-a)
                    diffs_A.add(a-x)
                sums_A.add(2*x)
                A.append(x)
            elif can_B:
                for b in B:
                    sums_B.add(x+b)
                    diffs_B.add(x-b)
                    diffs_B.add(b-x)
                sums_B.add(2*x)
                B.append(x)

        return A, B

    def local_search(A, B, deadline):
        A = list(A)
        B = list(B)
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)

        best_size = len(A) + len(B)
        best_A, best_B = A[:], B[:]
        all_elements = list(range(1, n+1))

        while time.time() < deadline:
            improved = False

            # Try adding elements
            for x in all_elements:
                if x not in A and can_add_to_A(x, A, sums_A, diffs_B):
                    for a in A:
                        sums_A.add(x+a)
                        diffs_A.add(x-a)
                        diffs_A.add(a-x)
                    sums_A.add(2*x)
                    A.append(x)
                    best_size = len(A) + len(B)
                    best_A, best_B = A[:], B[:]
                    improved = True
                elif x not in B and can_add_to_B(x, B, sums_B, diffs_A):
                    for b in B:
                        sums_B.add(x+b)
                        diffs_B.add(x-b)
                        diffs_B.add(b-x)
                    sums_B.add(2*x)
                    B.append(x)
                    best_size = len(A) + len(B)
                    best_A, best_B = A[:], B[:]
                    improved = True

            # Try moving elements between sets
            for x in list(A):
                if x not in B:
                    if can_add_to_B(x, B, sums_B, diffs_A):
                        A.remove(x)
                        sums_A = compute_sums(A)
                        diffs_A = compute_diffs(A)
                        for b in B:
                            sums_B.add(x+b)
                            diffs_B.add(x-b)
                            diffs_B.add(b-x)
                        sums_B.add(2*x)
                        B.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True

            for x in list(B):
                if x not in A:
                    if can_add_to_A(x, A, sums_A, diffs_B):
                        B.remove(x)
                        sums_B = compute_sums(B)
                        diffs_B = compute_diffs(B)
                        for a in A:
                            sums_A.add(x+a)
                            diffs_A.add(x-a)
                            diffs_A.add(a-x)
                        sums_A.add(2*x)
                        A.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True

            if not improved:
                break

        return best_A, best_B

    def polish(A, B, deadline):
        """Final high-precision polish: try all possible additions and swaps."""
        A = list(A)
        B = list(B)
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)
        best_size = len(A) + len(B)
        best_A, best_B = A[:], B[:]
        all_elements = list(range(1, n+1))

        while time.time() < deadline:
            improved = False
            # Multiple passes of adding
            for _ in range(3):
                for x in all_elements:
                    if x not in A and can_add_to_A(x, A, sums_A, diffs_B):
                        for a in A:
                            sums_A.add(x+a)
                            diffs_A.add(x-a)
                            diffs_A.add(a-x)
                        sums_A.add(2*x)
                        A.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True
                    elif x not in B and can_add_to_B(x, B, sums_B, diffs_A):
                        for b in B:
                            sums_B.add(x+b)
                            diffs_B.add(x-b)
                            diffs_B.add(b-x)
                        sums_B.add(2*x)
                        B.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True
                if not improved:
                    break

            # Try swaps: remove one from A, try to add something else
            if time.time() < deadline:
                for x in list(A):
                    if len(A) > 1:
                        A.remove(x)
                        sums_A = compute_sums(A)
                        diffs_A = compute_diffs(A)
                        # Try to add a new element
                        for y in all_elements:
                            if y not in A and y != x and can_add_to_A(y, A, sums_A, diffs_B):
                                for a in A:
                                    sums_A.add(y+a)
                                    diffs_A.add(y-a)
                                    diffs_A.add(a-y)
                                sums_A.add(2*y)
                                A.append(y)
                                if len(A)+len(B) > best_size:
                                    best_size = len(A) + len(B)
                                    best_A, best_B = A[:], B[:]
                                improved = True
                                break
                        # If no improvement, put x back
                        if len(A) + len(B) <= best_size - 1:
                            for a in A:
                                sums_A.add(x+a)
                                diffs_A.add(x-a)
                                diffs_A.add(a-x)
                            sums_A.add(2*x)
                            A.append(x)
                            sums_A = compute_sums(A)
                            diffs_A = compute_diffs(A)

                for x in list(B):
                    if len(B) > 1:
                        B.remove(x)
                        sums_B = compute_sums(B)
                        diffs_B = compute_diffs(B)
                        for y in all_elements:
                            if y not in B and y != x and can_add_to_B(y, B, sums_B, diffs_A):
                                for b in B:
                                    sums_B.add(y+b)
                                    diffs_B.add(y-b)
                                    diffs_B.add(b-y)
                                sums_B.add(2*y)
                                B.append(y)
                                if len(A)+len(B) > best_size:
                                    best_size = len(A) + len(B)
                                    best_A, best_B = A[:], B[:]
                                improved = True
                                break
                        if len(A) + len(B) <= best_size - 1:
                            for b in B:
                                sums_B.add(x+b)
                                diffs_B.add(x-b)
                                diffs_B.add(b-x)
                            sums_B.add(2*x)
                            B.append(x)
                            sums_B = compute_sums(B)
                            diffs_B = compute_diffs(B)

            if not improved:
                break

        return best_A, best_B

    best_A = []
    best_B = []
    best_total = 0

    for restart in range(300):
        if time.time() - start_time > time_limit:
            break
        random.seed(restart * 31 + 17)
        order = list(range(1, n+1))
        random.shuffle(order)

        A, B = run_greedy_balanced(order)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A, best_B = A[:], B[:]

        # Use remaining time for local search
        remaining = time_limit - (time.time() - start_time)
        if remaining > 0.5:
            A2, B2 = local_search(A, B, time.time() + remaining * 0.7)
            total2 = len(A2) + len(B2)
            if total2 > best_total:
                best_total = total2
                best_A, best_B = A2[:], B2[:]

    # Final polish pass with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        A3, B3 = polish(best_A, best_B, time.time() + remaining)
        total3 = len(A3) + len(B3)
        if total3 > best_total:
            best_total = total3
            best_A, best_B = A3[:], B3[:]

    return [best_A, best_B]
