import time

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    P = 101
    D = 10
    elements = list(range(1, D + 1))
    
    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)
    
    # Enumerate all nonempty subsets, group by size
    # subsets_by_size[k] = list of (residue, tuple_of_elements)
    subsets_by_size = [[] for _ in range(D + 1)]
    
    for mask in range(1, 1 << D):
        res = 0
        elems = []
        for i in range(D):
            if mask & (1 << i):
                x = i + 1
                res = (res + inv[x]) % P
                elems.append(x)
        sz = len(elems)
        subsets_by_size[sz].append((res, tuple(elems)))
    
    # For each residue, find the shortest subset that sums to it
    best_for_residue = [None] * P  # best_for_residue[r] = (size, tuple) or None
    
    for k in range(1, D + 1):
        if time.time() - t0 > deadline:
            break
        for res, elems in subsets_by_size[k]:
            if best_for_residue[res] is None:
                best_for_residue[res] = (k, elems)
    
    # Build result: for each residue, use the shortest subset found
    result = [None] * P
    max_len = 0
    for r in range(P):
        if best_for_residue[r] is not None:
            k, elems = best_for_residue[r]
            result[r] = list(elems)
            if k > max_len:
                max_len = k
        else:
            # Should not happen with 1023 subsets and 101 residues
            result[r] = [1]  # fallback
    
    return result
