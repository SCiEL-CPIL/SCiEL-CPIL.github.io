import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    
    def is_valid_with_x(S, sums_by_k, x):
        L = len(S)
        # new_sums[k] will hold sums of size k for S U {x}
        # sums_by_k[k] holds sums of size k for S
        new_sums = [set() for _ in range(L + 2)] # indices 0 to L+1
        
        # k=0 is always {0}
        new_sums[0] = {0}
        
        for k in range(1, L + 2):
            # Sums of size k in S U {x} come from:
            # 1. Sums of size k in S (if k <= L)
            # 2. x + sum of size k-1 in S
            if k <= L:
                new_sums[k].update(sums_by_k[k])
            
            # Add x + sums of size k-1 from S
            # sums_by_k[k-1] exists for k-1 <= L
            for s in sums_by_k[k-1]:
                new_sums[k].add(x + s)
        
        # Check disjointness of new_sums for different k
        sum_to_k = {}
        for k in range(L + 2):
            for s in new_sums[k]:
                if s in sum_to_k:
                    if sum_to_k[s] != k:
                        return False, None
                else:
                    sum_to_k[s] = k
        
        return True, new_sums

    def greedy_build(order):
        S = []
        sums_by_k = [set([0])]
        
        for x in order:
            if time.time() > deadline:
                break
            
            valid, new_sums = is_valid_with_x(S, sums_by_k, x)
            if valid:
                S.append(x)
                sums_by_k = new_sums
        
        return S

    # Strategy 1: Smallest first
    set1 = greedy_build(range(1, 257))
    best_set = set1 if len(set1) > len(best_set) else best_set
    
    # Strategy 2: Random restarts
    # We can try a few random orders
    # To be safe, we limit the number of restarts based on time
    max_restarts = 20
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        
        order = list(range(1, 257))
        random.shuffle(order)
        # Bias towards smaller numbers? Or just random.
        # Maybe sort by some heuristic?
        # For now, random.
        
        s = greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
            
    # Ensure types are correct
    return [int(x) for x in best_set]
