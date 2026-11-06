import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length by preferring shorter subsets.
    """
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    P = 101
    D = 10
    elements = list(range(1, D + 1))

    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)

    # Enumerate all nonempty subsets using bitmask, compute residue and size
    # Store as: subsets_by_size[size].append((residue, list_of_elements))
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
        subsets_by_size[sz].append((res, elems))

    # Greedily assign shortest subset to each residue
    # Process by increasing length
    results = [None] * P
    covered_count = 0

    for length in range(1, D + 1):
        if time.time() - t0 > deadline:
            break
        for res, elems in subsets_by_size[length]:
            if results[res] is None:
                results[res] = elems
                covered_count += 1

        # Check if all residues are covered
        if covered_count == P:
            break

    # If any residue is still None (shouldn't happen), fill with any valid subset
    if covered_count < P:
        for mask in range(1, 1 << D):
            if time.time() - t0 > deadline:
                break
            res = 0
            elems = []
            for i in range(D):
                if mask & (1 << i):
                    x = i + 1
                    res = (res + inv[x]) % P
                    elems.append(x)
            if results[res] is None:
                results[res] = elems
                covered_count += 1
            if covered_count == P:
                break

    # Last resort: if still not covered (impossible), use a default
    if covered_count < P:
        for r in range(P):
            if results[r] is None:
                results[r] = [1]

    return results
