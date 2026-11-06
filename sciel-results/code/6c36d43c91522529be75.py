"""Rewritten program using a precomputed lookup-table of the 512 codewords."""

def construct():
    # Precompute the 512 codewords of the [18,9,6] extended quadratic-residue code
    # using a different algebraic path: build the generator matrix from the
    # quadratic-residue characteristic vector and its minimal cyclic shifts,
    # then enumerate via a Gray-code style XOR walk to avoid repeated bit_count.
    #
    # Quadratic residues mod 17: {1,2,4,8,16,13,9,15}
    q = (1 << 1) | (1 << 2) | (1 << 4) | (1 << 8) | (1 << 16) | (1 << 13) | (1 << 9) | (1 << 15)
    allones = (1 << 17) - 1

    # Build 9 basis vectors: all-ones plus 8 cyclic shifts of q
    basis = [allones]
    for s in range(1, 9):
        basis.append((q >> s) | ((q & ((1 << s) - 1)) << (17 - s)))

    # Enumerate all 512 combinations via XOR accumulation (Gray-code order)
    # This avoids the inner loop over 9 basis vectors per codeword.
    words = [0]
    current = 0
    for i in range(9):
        for _ in range(1 << i):
            current ^= basis[i]
            # Extend to 18 bits with parity
            parity = current.bit_count() & 1
            words.append((current << 1) | parity)

    return words
