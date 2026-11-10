import time

def construct():
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0
    
    # Precompute modular inverses for denominators 1..10
    P = 101
    N = 10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
        
    # Precompute all subsets
    all_subsets = []
    for mask in range(1, 1 << N):
        length = bin(mask).count('1')
        s = 0
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                denoms.append(d)
                s = (s + inv[d]) % P
        all_subsets.append((length, s, denoms))
        
    # Sort by length to process smallest first
    all_subsets.sort(key=lambda x: x[0])
    
    # Find shortest subset for each residue
    assignment = [None] * P
    is_covered = [False] * P
    covered = 0
    
    for length, r, denoms in all_subsets:
        if not is_covered[r]:
            assignment[r] = denoms
            is_covered[r] = True
            covered += 1
            
            if covered == P:
                break
                
    # Fallback for any uncovered residues (should not happen)
    for r in range(P):
        if assignment[r] is None:
            for l, res, d in all_subsets:
                if res == r:
                    assignment[r] = d
                    break
            if assignment[r] is None:
                assignment[r] = [1]
        
    return assignment

# EVOLVE-BLOCK-END
