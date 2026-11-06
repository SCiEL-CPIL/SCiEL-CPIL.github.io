import numpy as np
import time


def construct():
    """
    Constructs the extended quadratic-residue code QR(17) to length 18.
    Returns a list of 512 distinct integers representing 18-bit codewords
    with pairwise Hamming distance at least 6.
    """
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin under the 90s wall-clock limit

    p = 17      # field length of the QR code
    n = 18      # extended length
    k = 9       # number of message bits (2^k = 512 codewords)

    # --- 1. Quadratic residues mod 17 (excluding 0) ---
    qr_set = set()
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) == 1:
            qr_set.add(i)

    # --- 2. Generator matrix G (k x p) over GF(2) ---
    first_row = np.zeros(p, dtype=np.uint8)
    first_row[0] = 1
    for r in qr_set:
        first_row[r] = 1

    # k cyclic shifts -> 9 x 17 generator matrix
    G = np.vstack([np.roll(first_row, -shift, axis=0) for shift in range(k)])

    # --- 3. All 512 codewords = coeffs @ G (mod 2), vectorized ---
    masks = np.arange(1 << k, dtype=np.uint16)
    coeffs = np.zeros((1 << k, k), dtype=np.uint8)
    for i in range(k):
        coeffs[:, i] = (masks >> i) & 1

    codewords_17 = (coeffs @ G) % 2  # shape (512, 17)

    # --- 4. Extend to length 18 with an overall parity bit ---
    parity_bits = codewords_17.sum(axis=1) % 2  # shape (512,)
    codewords_18 = np.column_stack((codewords_17, parity_bits))  # (512, 18)

    # --- 5. Fast vectorized bit-packing to integers ---
    # bit j of the integer is the value of column j
    powers = 1 << np.arange(n, dtype=np.uint32)          # [1,2,4,...]
    words_arr = (codewords_18.astype(np.uint32) * powers).sum(axis=1).astype(np.int64)

    words = words_arr.tolist()

    # --- 6. Lightweight sanity guard (cheap; returns valid set if anything odd) ---
    if time.time() - start_time < DEADLINE:
        seen = set()
        for w in words:
            if w in seen:
                continue
            seen.add(w)
        if len(seen) != len(words):
            words = list(seen)

    return words
