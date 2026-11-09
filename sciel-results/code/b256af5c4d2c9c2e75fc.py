import time
import numpy as np
from itertools import combinations

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended QR(17) code: [17, 8, 5] -> [18, 8, 6] with 256 words
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # 0-indexed positions: {0, 1, 3, 7, 8, 12, 14, 15}
    row0 = np.zeros(17, dtype=np.uint8)
    for pos in [0, 1, 3, 7, 8, 12, 14, 15]:
        row0[pos] = 1

    # Build 8x17 generator matrix (circulant)
    G = np.zeros((8, 17), dtype=np.uint8)
    for i in range(8):
        G[i] = np.roll(row0, i)

    # Compute all 2^8 = 256 codewords of QR(17)
    qr_words_17 = []
    for mask in range(256):
        bits = [(mask >> i) & 1 for i in range(8)]
        word = 0
        for i in range(8):
            if bits[i]:
                word ^= int(np.packbits(G[i], bitorder='msb')[0]) if False else 0
        # Simpler: XOR rows directly
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                for j in range(17):
                    if G[i][j]:
                        word ^= (1 << j)
        qr_words_17.append(word)

    # Extend to length 18 with parity bit
    qr_words_18 = []
    for w in qr_words_17:
        parity = w.bit_count() % 2
        extended = w | (parity << 17)
        qr_words_18.append(extended)

    # Verify minimum distance
    min_d = N
    for i in range(len(qr_words_18)):
        for j in range(i + 1, len(qr_words_18)):
            d = (qr_words_18[i] ^ qr_words_18[j]).bit_count()
            if d < min_d:
                min_d = d

    if min_d < MIN_DIST:
        # Fallback: Reed-Muller [16,5,8] -> extend to [18,5,8] with 32 words
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        # Pad to 18 bits (already 16 bits, add 2 zero bits)
        current_words = [w << 2 for w in rm_words]
    else:
        current_words = list(qr_words_18)

    current_set = set(current_words)
    current_arr = np.array(current_words, dtype=np.uint32)

    # Vectorized greedy: check candidates in batches using NumPy
    # Generate all 2^18 candidates, filter out existing
    all_candidates = np.array([c for c in range(1 << N) if c not in current_set], dtype=np.uint32)

    batch_size = 2048
    for start in range(0, len(all_candidates), batch_size):
        if time.time() > deadline - 5.0:
            break

        batch = all_candidates[start:start + batch_size]
        if len(batch) == 0:
            break

        # Compute XOR matrix: (batch_len, cur_len)
        xor_matrix = batch[:, np.newaxis] ^ current_arr[np.newaxis, :]

        # Compute popcount using bit unpacking
        # Convert to binary representation
        bits = np.zeros((len(batch), len(current_arr), N), dtype=np.uint8)
        for b in range(N):
            bits[:, :, b] = (xor_matrix >> b) & 1

        popcounts = bits.sum(axis=2)
        min_pops = popcounts.min(axis=1)

        valid_indices = np.where(min_pops >= MIN_DIST)[0]

        for idx in valid_indices:
            candidate = int(batch[idx])
            current_words.append(candidate)
            current_set.add(candidate)
            current_arr = np.append(current_arr, candidate)

    return current_words
