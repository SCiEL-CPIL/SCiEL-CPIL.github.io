import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    p = 1009

    # Precompute quadratic residues and store one root for each
    # Using a list for O(1) lookup; root_of[i] = r where r*r % p == i, or -1
    root_of = [-1] * p
    is_residue = [False] * p

    for r in range(1, p):
        res = (r * r) % p
        if root_of[res] == -1:
            root_of[res] = r
        if res != 0:  # zero residue excluded
            is_residue[res] = True

    # Find the smallest s in [1, 1004] such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod 1009.
    for s in range(1, 1005):
        if time.time() - start_time > deadline:
            break

        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2]
                and is_residue[s + 3] and is_residue[s + 4]):
            roots = [root_of[s], root_of[s + 1], root_of[s + 2],
                     root_of[s + 3], root_of[s + 4]]
            # Verify each root satisfies ri**2 mod 1009 == s+i
            for i, ri in enumerate(roots):
                if (ri * ri) % p != (s + i):
                    raise RuntimeError(f"Root verification failed at index {i}")
            return {
                'start': s,
                'roots': roots
            }

    # Fallback (should not be reached for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
