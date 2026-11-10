import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0
    
    P = 101
    N = 10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
        
    # Precompute all subsets with their length and sum
    # We want to find the minimum length for each residue.
    min_len_for_residue = [float('inf')] * P
    best_subset_for_residue = [None] * P
    
    # Iterate all masks
    for mask in range(1, 1 << N):
        length = bin(mask).count('1')
        s = 0
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                denoms.append(d)
                s = (s + inv[d]) % P
        
        # Update if this length is shorter than the current best for this residue
        if length < min_len_for_residue[s]:
            min_len_for_residue[s] = length
            best_subset_for_residue[s] = denoms
            
    # Construct the assignment
    assignment = [None] * P
    max_len = 0
    for r in range(P):
        if best_subset_for_residue[r] is not None:
            assignment[r] = best_subset_for_residue[r]
            if len(assignment[r]) > max_len:
                max_len = len(assignment[r])
        else:
            # Fallback, should not happen
            assignment[r] = [1]
            
    return assignment
