import time

def construct():
    p = 1009
    deadline = time.time() + 85

    # Precompute the set of non-zero quadratic residues and a root for each.
    # For each x in 1..p-1, x^2 mod p is a QR. We store the smallest root found.
    is_qr = [False] * p
    roots = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if roots[r] == 0:
                roots[r] = x

    # Use a bitset approach to find the first run of 5 consecutive QRs.
    # Bit i is 1 if i is a QR (for i in 1..p-1).
    bitset = 0
    for i in range(1, p):
        if is_qr[i]:
            bitset |= (1 << i)

    # We want the smallest s in [1, 1004] such that bits s, s+1, s+2, s+3, s+4 are all set.
    # Create a mask with 5 consecutive 1-bits starting at position 0: 0b11111
    mask5 = 0b11111

    # Shift the bitset right by 1 so that bit 0 corresponds to index 1 in the original.
    # Actually, let's think differently. 
    # If bit i is set, then i is a QR.
    # We want s such that bits s, s+1, s+2, s+3, s+4 are set.
    # This is equivalent to: (bitset >> s) & mask5 == mask5.
    # We want the smallest s in [1, 1004].

    min_s = -1
    for s in range(1, 1005):
        if time.time() > deadline - 1:
            break
        if (bitset >> s) & mask5 == mask5:
            min_s = s
            break

    if min_s == -1:
        # Fallback to simple loop
        for s in range(1, 1005):
            if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
                min_s = s
                break

    if min_s == -1:
        raise RuntimeError("No run of 5 consecutive QRs found")

    result_roots = []
    for i in range(5):
        r = min_s + i
        if roots[r] == 0:
            raise RuntimeError(f"No root found for residue {r}")
        result_roots.append(roots[r])

    return {'start': min_s, 'roots': result_roots}
