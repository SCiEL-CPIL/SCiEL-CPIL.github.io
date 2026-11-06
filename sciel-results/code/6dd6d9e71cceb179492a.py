# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    best_A = []
    best_B = []
    best_size = 0
    
    def can_add_A(x, A, sums_A, diffs_A, diffs_B):
        # Check Sidon condition for A: unique sums a_i + a_j (i<=j)
        # New sums: 2*x, x+a for a in A
        # New diffs: |x-a| for a in A
        # Must not be in sums_A (except new ones being added, but we check before adding)
        # Actually, if x+a is in sums_A, it's a conflict.
        # If 2*x is in sums_A, conflict.
        # If |x-a| is in diffs_A or diffs_B, conflict.
        
        if 2 * x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            d = abs(x - a)
            if d in diffs_A or d in diffs_B:
                return False
        return True

    def can_add_B(x, B, sums_B, diffs_B, diffs_A):
        if 2 * x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            d = abs(x - b)
            if d in diffs_B or d in diffs_A:
                return False
        return True

    def run_greedy(order, strategy):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            added = False
            
            if strategy == 0: # Try A first
                if can_add_A(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    sums_A.add(2 * x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        d = abs(x - a)
                        diffs_A.add(d)
                    added = True
                if not added and can_add_B(x, B, sums_B, diffs_B, diffs_A):
                    B.append(x)
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        d = abs(x - b)
                        diffs_B.add(d)
                        added = True
            elif strategy == 1: # Try B first
                if can_add_B(x, B, sums_B, diffs_B, diffs_A):
                    B.append(x)
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        d = abs(x - b)
                        diffs_B.add(d)
                    added = True
                if not added and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    sums_A.add(2 * x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        d = abs(x - a)
                        diffs_A.add(d)
                        added = True
            else: # Random choice
                if random.random() < 0.5:
                    if can_add_A(x, A, sums_A, diffs_A, diffs_B):
                        A.append(x)
                        sums_A.add(2 * x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d = abs(x - a)
                            diffs_A.add(d)
                        added = True
                    if not added and can_add_B(x, B, sums_B, diffs_B, diffs_A):
                        B.append(x)
                        sums_B.add(2 * x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d = abs(x - b)
                            diffs_B.add(d)
                            added = True
                else:
                    if can_add_B(x, B, sums_B, diffs_B, diffs_A):
                        B.append(x)
                        sums_B.add(2 * x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d = abs(x - b)
                            diffs_B.add(d)
                        added = True
                    if not added and can_add_A(x, A, sums_A, diffs_A, diffs_B):
                        A.append(x)
                        sums_A.add(2 * x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d = abs(x - a)
                            diffs_A.add(d)
                            added = True
        
        return A, B

    # Try a few deterministic orders first
    # 1. Ascending, A first
    A, B = run_greedy(list(range(1, 257)), 0)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A, best_B = A, B

    # 2. Descending, B first
    A, B = run_greedy(list(range(256, 0, -1)), 1)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A, best_B = A, B
        
    # 3. Ascending, B first
    A, B = run_greedy(list(range(1, 257)), 1)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A, best_B = A, B

    # 4. Descending, A first
    A, B = run_greedy(list(range(256, 0, -1)), 0)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A, best_B = A, B

    # Random restarts
    while time.time() - start_time < time_limit:
        random.seed()
        order = list(range(1, 257))
        random.shuffle(order)
        strategy = random.randint(0, 2)
        A, B = run_greedy(order, strategy)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A, best_B = A, B

    return [best_A, best_B]
# EVOLVE-BLOCK-END
