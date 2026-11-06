"""New approach: Randomized Greedy with Restarts for two Sidon sets."""
import random
import time

def construct():
    n = 256
    max_size = 64
    
    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        def can_add_A(x):
            # Sidon check: new sums must not collide with existing sums
            for a in A:
                if (a + x) in sums_A:
                    return False
            if (2 * x) in sums_A:
                return False
            
            # Disjoint difference check: new diffs must not be in diffs_B
            # Note: If sum check passes, new diffs are guaranteed unique within A
            for a in A:
                d = x - a
                if d != 0 and d in diffs_B:
                    return False
            return True

        def can_add_B(x):
            for b in B:
                if (b + x) in sums_B:
                    return False
            if (2 * x) in sums_B:
                return False
            
            for b in B:
                d = x - b
                if d != 0 and d in diffs_A:
                    return False
            return True

        def add_to_A(x):
            for a in A:
                sums_A.add(a + x)
                d = x - a
                if d != 0:
                    diffs_A.add(d)
                    diffs_A.add(-d)
            sums_A.add(2 * x)
            A.append(x)

        def add_to_B(x):
            for b in B:
                sums_B.add(b + x)
                d = x - b
                if d != 0:
                    diffs_B.add(d)
                    diffs_B.add(-d)
            sums_B.add(2 * x)
            B.append(x)

        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            a_ok = len(A) < max_size and can_add_A(x)
            b_ok = len(B) < max_size and can_add_B(x)
            
            if a_ok and b_ok:
                # Heuristic: add to the smaller set to balance
                if len(A) <= len(B):
                    add_to_A(x)
                else:
                    add_to_B(x)
            elif a_ok:
                add_to_A(x)
            elif b_ok:
                add_to_B(x)
                
        return A, B

    best_score = 0
    best_A = []
    best_B = []
    
    # Baseline: deterministic order
    A, B = greedy_construct(list(range(1, n + 1)))
    score = len(A) + len(B)
    if score > best_score:
        best_score = score
        best_A, best_B = A, B
        
    # Randomized restarts
    start_time = time.time()
    # Estimate: ~2000 restarts should be safe within 60s for Python
    for i in range(2000):
        if time.time() - start_time > 55:
            break
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B = greedy_construct(order)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A, B
            
    return [best_A, best_B]
