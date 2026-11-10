import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check Sidon property for new sums in S
        if 2 * x in sums_S:
            return False
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

    def local_improve(A, B):
        """Try to improve by adding remaining candidates or swapping"""
        nonlocal best_size, best_A, best_B
        A = A[:]
        B = B[:]
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Rebuild state
        for a in A:
            sums_A.add(2 * a)
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                sums_A.add(A[i] + A[j])
                diffs_A.add(abs(A[i] - A[j]))
        for b in B:
            sums_B.add(2 * b)
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                sums_B.add(B[i] + B[j])
                diffs_B.add(abs(B[i] - B[j]))
        
        # Try to add any candidate that fits
        for x in range(1, n + 1):
            if time.time() - start_time > time_limit:
                break
            if x not in A and x not in B:
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
        
        return A, B

    # Strategy: Use smaller-first bias for better Sidon packing
    # Try multiple orderings and local improvements
    
    restarts = 0
    while time.time() - start_time < time_limit:
        # Create a biased order: smaller numbers first with some randomization
        order = list(range(1, n + 1))
        # Shuffle but bias towards smaller numbers
        for i in range(len(order) - 1):
            # With decreasing probability, swap with a later element
            if random.random() < 0.3 * (1 - i / len(order)):
                j = random.randint(i, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(order, prefer_A)
        
        # Local improvement pass
        A, B = local_improve(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
            if best_size >= 32:
                # Keep searching a bit more to see if we can do better
                pass
        
        restarts += 1
        # Adaptive: if stuck, try more aggressive randomization
        if restarts % 50 == 0 and best_size < 30:
            # Increase randomization
            pass

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
