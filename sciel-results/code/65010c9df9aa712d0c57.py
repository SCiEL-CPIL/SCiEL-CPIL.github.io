"""Rewritten program for the suite task; conflict-graph + greedy/local search approach."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    max_size = 64
    rng = random.Random(42)
    
    # We'll try multiple random seeds and keep the best result
    best_total = 0
    best_A = []
    best_B = []
    
    def is_sidon(sets):
        """Check if all pair sums (with replacement) are unique."""
        sums = set()
        for i, a in enumerate(sets):
            for j in range(i, len(sets)):
                b = sets[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(sets):
        """Get nonzero differences for a set."""
        diffs = set()
        for i in range(len(sets)):
            for j in range(len(sets)):
                if i != j:
                    d = sets[j] - sets[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def valid_solution(A, B):
        """Check if A and B form a valid solution."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        if not is_sidon(A) or not is_sidon(B):
            return False
        diffA = get_diffs(A)
        diffB = get_diffs(B)
        if diffA & diffB:
            return False
        return True
    
    def greedy_construct(order_A, order_B, A, B):
        """Greedy add elements in given order."""
        A = list(A)
        B = list(B)
        
        for x in order_A:
            if len(A) >= max_size:
                break
            candidate = A + [x]
            if is_sidon(candidate):
                # Check difference disjointness
                diffA = get_diffs(candidate)
                diffB = get_diffs(B)
                if not (diffA & diffB):
                    A = candidate
        
        for x in order_B:
            if len(B) >= max_size:
                break
            candidate = B + [x]
            if is_sidon(candidate):
                diffB = get_diffs(candidate)
                diffA = get_diffs(A)
                if not (diffA & diffB):
                    B = candidate
        
        return A, B
    
    def local_search(A, B, max_iters=5000):
        """Local search to improve total size."""
        A = list(A)
        B = list(B)
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding an element to A
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A:
                        candidate = A + [x]
                        if is_sidon(candidate):
                            diffA = get_diffs(candidate)
                            diffB = get_diffs(B)
                            if not (diffA & diffB):
                                A = candidate
                                improved = True
                                break
            
            # Try adding an element to B
            if len(B) < max_size:
                for x in range(1, n + 1):
                    if x not in B:
                        candidate = B + [x]
                        if is_sidon(candidate):
                            diffB = get_diffs(candidate)
                            diffA = get_diffs(A)
                            if not (diffA & diffB):
                                B = candidate
                                improved = True
                                break
            
            # Try swapping: remove from A, add to B (or vice versa)
            if not improved and len(A) > 0 and len(B) < max_size:
                for i, a in enumerate(A):
                    A_without = A[:i] + A[i+1:]
                    for x in range(1, n + 1):
                        if x not in B:
                            B_candidate = B + [x]
                            if is_sidon(B_candidate):
                                diffB_new = get_diffs(B_candidate)
                                diffA_new = get_diffs(A_without)
                                if not (diffA_new & diffB_new):
                                    if len(B_candidate) + len(A_without) > len(B) + len(A):
                                        A = A_without
                                        B = B_candidate
                                        improved = True
                                        break
                    if improved:
                        break
            
            if not improved and len(B) > 0 and len(A) < max_size:
                for i, b in enumerate(B):
                    B_without = B[:i] + B[i+1:]
                    for x in range(1, n + 1):
                        if x not in A:
                            A_candidate = A + [x]
                            if is_sidon(A_candidate):
                                diffA_new = get_diffs(A_candidate)
                                diffB_new = get_diffs(B_without)
                                if not (diffA_new & diffB_new):
                                    if len(A_candidate) + len(B_without) > len(A) + len(B):
                                        A = A_candidate
                                        B = B_without
                                        improved = True
                                        break
                    if improved:
                        break
            
            if not improved:
                break
        
        return A, B
    
    # Try multiple random orderings
    for seed in range(20):
        rng.seed(seed)
        
        # Create random orderings
        all_elements = list(range(1, n + 1))
        order_A = all_elements[:]
        order_B = all_elements[:]
        rng.shuffle(order_A)
        rng.shuffle(order_B)
        
        A, B = greedy_construct(order_A, order_B, [], [])
        
        # Local search
        A, B = local_search(A, B, max_iters=2000)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
        
        # Early exit if we hit the theoretical max
        if best_total >= 128:
            break
    
    # Final validation
    if not valid_solution(best_A, best_B):
        # Fallback to simple construction
        A = list(range(1, 17))
        B = list(range(17, 33))
        if not valid_solution(A, B):
            A = [1, 2, 4, 8, 16, 32, 64, 128]
            B = [3, 5, 9, 17, 33, 65, 129, 257] if 257 <= n else [3, 5, 9, 17, 33, 65, 129]
            B = [x for x in B if x <= n]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
