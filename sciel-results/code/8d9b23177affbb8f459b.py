import time

"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""

# EVOLVE-BLOCK-START
def construct():
    """
    Return {'start': s, 'roots': [r0, r1, r2, r3, r4]} where:
      - s is the smallest integer in [1, 1004] such that s, s+1, ..., s+4
        are all non-zero quadratic residues modulo 1009.
      - roots[i] is a root r_i in [1, 1008] with r_i**2 % 1009 == s+i.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep margin

    p = 1009
    m = 5

    # Precompute roots and QR markers
    root_of = [0] * p
    is_qr = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            root_of[r] = x
            is_qr[r] = 1

    # Compute prefix sums for O(1) range-sum queries
    prefix = [0] * (p + 1)
    for i in range(1, p):
        prefix[i + 1] = prefix[i] + is_qr[i]

    # Find minimal s
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        if prefix[s + m] - prefix[s] == m:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback if deadline hit or not found (should not happen for p=1009)
    # Return best effort or raise error
    # Since p=1009 is small, this loop is fast.
    # If we get here, it means no run was found or time ran out.
    # For safety, return a valid known run if possible, or raise.
    # Given the math, s=1 is valid.
    # If the loop failed, something is wrong with the environment.
    raise RuntimeError('No run of 5 consecutive QRs found or timeout')
# EVOLVE-BLOCK-END
