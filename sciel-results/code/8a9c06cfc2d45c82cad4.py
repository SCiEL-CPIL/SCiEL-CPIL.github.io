"""Find the minimal start s for a run of 5 quadratic residues mod 1009."""

def construct():
    p = 1009
    m = 5

    # Precompute a mapping from non-zero quadratic residues to a valid root in [1, p-1]
    # Using a list for O(1) lookup
    roots = [0] * p
    is_res = [False] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_res[r]:
                roots[r] = x
                is_res[r] = True

    # Search for the minimal s in [1, 1004]
    # We need s, s+1, s+2, s+3, s+4 to be non-zero quadratic residues.
    # Max s is 1004 because s+4 <= 1008 (since residues are in 1..1008, 0 is excluded)
    for s in range(1, 1005):
        # Check if s+i is a valid residue for i in 0..4
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(m)]
            }

    # This should never happen for p=1009, but raise an error just in case
    raise RuntimeError("No valid run found")
