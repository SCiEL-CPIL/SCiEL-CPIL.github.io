"""Rewritten program for the suite task; incremental state greedy + local search."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    max_size = 64
    rng = random.Random(42)
    
    best_total = 0
    best_A = []
    best_B = []
    
    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if (x - a) in diffs_B:
                return False
            if (a - x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if (x - b) in diffs_A:
                return False
            if (b - x) in diffs_A:
                return False
        return True

    def add_to_A(x, A, sums_A, diffs_A):
        A.append(x)
        sums_A.add(x + x)
        for a in A[:-1]:
            sums_A.add(x + a)
            diffs_A.add(x - a)
            diffs_A.add(a - x)
            
    def add_to_B(x, B, sums_B, diffs_B, diffs_A):
        B.append(x)
        sums_B.add(x + x)
        for b in B[:-1]:
            sums_B.add(x + b)
            diffs_B.add(x - b)
            diffs_B.add(b - x)

    def rebuild_state(A, B):
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
                    
        return sums_A, diffs_A, sums_B, diffs_B

    def valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
        if diffs_A & diffs_B:
            return False
        return True

    def greedy_construct(order_A, order_B, a_first=True):
        A = []
        B = []
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        if a_first:
            for x in order_A:
                if len(A) >= max_size: break
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    add_to_A(x, A, sums_A, diffs_A)
            for x in order_B:
                if len(B) >= max_size: break
                if can_add_to_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
                    add_to_B(x, B, sums_B, diffs_B, diffs_A)
        else:
            for x in order_B:
                if len(B) >= max_size: break
                if can_add_to_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
                    add_to_B(x, B, sums_B, diffs_B, diffs_A)
            for x in order_A:
                if len(A) >= max_size: break
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    add_to_A(x, A, sums_A, diffs_A)
                    
        return A, B

    def local_search(A, B, max_iters=500):
        A = list(A)
        B = list(B)
        
        for _ in range(max_iters):
            improved = False
            sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
            
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A:
                        if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                            add_to_A(x, A, sums_A, diffs_A)
                            improved = True
                            break
                            
            if len(B) < max_size:
                if improved:
                    sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
                for x in range(1, n + 1):
                    if x not in B:
                        if can_add_to_B(x, A, sums_A, diffs_A, diffs_B, B, sums_B):
                            add_to_B(x, B, sums_B, diffs_B, diffs_A)
                            improved = True
                            break

            if not improved:
                break
                
        return A, B

    for seed in range(50):
        rng.seed(seed)
        
        all_elements = list(range(1, n + 1))
        order_A = all_elements[:]
        order_B = all_elements[:]
        rng.shuffle(order_A)
        rng.shuffle(order_B)
        
        A1, B1 = greedy_construct(order_A, order_B, a_first=True)
        A2, B2 = greedy_construct(order_A, order_B, a_first=False)
        
        if len(A1) + len(B1) >= len(A2) + len(B2):
            A, B = A1, B1
        else:
            A, B = A2, B2
            
        A, B = local_search(A, B, max_iters=500)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
        if best_total >= 128:
            break

    if not valid_solution(best_A, best_B):
        A = list(range(1, 17))
        B = list(range(17, 33))
        if not valid_solution(A, B):
            A = [1, 2, 4, 8, 16, 32, 64, 128]
            B = [3, 5, 9, 17, 33, 65, 129]
            B = [x for x in B if x <= n]
            best_A = A
            best_B = B

    return [best_A, best_B]
# EVOLVE-BLOCK-END
