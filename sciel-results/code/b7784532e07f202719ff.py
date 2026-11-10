import time

def construct():
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0

    # Problem parameters
    P = 101
    N = 10
    
    # Precompute modular inverses for denominators 1..10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)

    # Precompute all non-empty subsets
    # Each entry: (length, residue_sum, denominators_list)
    all_subsets = []
    
    # Iterate through all non-empty masks
    for mask in range(1, 1 << N):
        length = 0
        s = 0
        denoms = []
        
        # Unpack mask to find selected denominators
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                denoms.append(d)
                s = (s + inv[d]) % P
                length += 1
                
        all_subsets.append((length, s, denoms))

    # Sort subsets by length (ascending) to ensure we pick the shortest for each residue first
    all_subsets.sort(key=lambda x: x[0])

    # Assignment array
    assignment = [None] * P
    is_covered = [False] * P
    covered = 0

    # Greedy assignment: Since choices are independent, picking the shortest 
    # subset for each residue individually minimizes the maximum row length.
    for length, r, denoms in all_subsets:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        if not is_covered[r]:
            assignment[r] = denoms
            is_covered[r] = True
            covered += 1
            
            if covered == P:
                break
                
    # Fallback for any uncovered residues (should not happen if coverage is complete)
    for r in range(P):
        if assignment[r] is None:
            # Find any subset that sums to r
            for l, res, d in all_subsets:
                if res == r:
                    assignment[r] = d
                    break
            if assignment[r] is None:
                # This should never happen for P=101, N=10
                assignment[r] = [1]

    return assignment
