# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin
    
    best_size = 0
    best_a = []
    best_b = []
    
    # Helper to check if x can be added to S (list) with sums S_sums and diffs S_diffs
    # against other set T with diffs T_diffs
    def can_add(x, S, S_sums, S_diffs, T_diffs):
        # Check Sidon property for S U {x}
        # 1. 2x not in S_sums
        if 2*x in S_sums:
            return False
        # 2. x+s not in S_sums for all s in S
        for s in S:
            if x + s in S_sums:
                return False
        
        # Check Disjoint Difference property with T
        # New diffs are |x - s| for s in S
        for s in S:
            d = abs(x - s)
            if d in T_diffs:
                return False
        return True

    def update_sets(x, S, S_sums, S_diffs):
        # Add x to S
        S.append(x)
        S_sums.add(2*x)
        for s in S[:-1]: # S[:-1] is the old S
            S_sums.add(x + s)
            S_diffs.add(abs(x - s))
        return S, S_sums, S_diffs

    # Run multiple randomized greedy passes
    while time.time() - start_time < time_limit:
        A = []
        B = []
        A_sums = set()
        B_sums = set()
        A_diffs = set()
        B_diffs = set()
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if can_add(x, A, A_sums, A_diffs, B_diffs):
                A, A_sums, A_diffs = update_sets(x, A, A_sums, A_diffs)
            # Try adding to B
            elif can_add(x, B, B_sums, B_diffs, A_diffs):
                B, B_sums, B_diffs = update_sets(x, B, B_sums, B_diffs)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_a = list(A)
            best_b = list(B)
            
            # Early exit if we hit a high value
            # Theoretical max for Sidon in [1, 256] is ~16. 
            # So 32 is a very good upper bound for sum.
            if best_size >= 32:
                break

    # Ensure non-empty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
        
    return [best_a, best_b]
# EVOLVE-BLOCK-END
