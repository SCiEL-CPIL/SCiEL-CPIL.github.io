def construct():
    """
    Find the minimum start s for a run of 5 consecutive non-zero quadratic residues mod 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    p = 1009
    m = 5

    # Precompute:
    #   is_qr[r] = True iff r is a non-zero quadratic residue mod p
    #   root_of[r] = smallest x in [1, p-1] with x*x % p == r (for QRs)
    is_qr = [False] * p
    root_of = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if root_of[r] == 0:
                root_of[r] = x

    # Search for the minimum s in ascending order.
    # s must satisfy 1 <= s and s+4 <= p-1, i.e. s <= p-5 = 1004.
    for s in range(1, p - m + 1):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No valid run of 5 consecutive non-zero quadratic residues found")
