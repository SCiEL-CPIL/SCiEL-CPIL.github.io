import time
import random

N = 18
MIN_DIST = 6
TIME_LIMIT = 90.0


def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    The known optimal value is A_2(18,6) = 512, achieved by the extended
    quadratic-residue code of length 18, a [18, 9, 6] linear code.

    This implementation builds the QR(17) cyclic shifts, finds a basis via
    fast span membership, generates all 512 codewords, extends with a parity
    bit, and then attempts a brief local search for any (extremely unlikely)
    improvement beyond 512.  If no improvement is found, the 512-word QR code
    is returned.
    """
    deadline = time.time() + TIME_LIMIT - 5.0
    p = 17
    qr = set((i * i) % p for i in range(p))

    # Build the 17 cyclic shifts of the QR indicator vector (17-bit words)
    shifts = []
    for k in range(p):
        w = 0
        for j in range(p):
            if (j - k) % p in qr:
                w |= (1 << j)
        shifts.append(w)

    # Find a basis of size 9
    basis = []
    for w in shifts:
        if not basis:
            basis.append(w)
            continue

        # Fast span membership via incremental set
        span = {0}
        for b in basis:
            new_items = {s ^ b for s in span}
            span |= new_items

        if w not in span:
            basis.append(w)
            if len(basis) == 9:
                break

    # Generate all 512 codewords, extended with parity bit
    words_17 = []
    for mask in range(512):
        w = 0
        for i in range(9):
            if mask & (1 << i):
                w ^= basis[i]
        words_17.append(w)

    words = [w | ((w.bit_count() % 2) << 17) for w in words_17]

    # Quick pairwise verification (130,656 pairs, fast in CPython)
    for i in range(len(words)):
        wi = words[i]
        for j in range(i + 1, len(words)):
            if (wi ^ words[j]).bit_count() < MIN_DIST:
                # Should never happen; return what we have
                return words

    # Attempt a brief local search for a larger code
    # Strategy: try to add a new word that is at distance >= 6 from all existing words
    # We sample candidate 18-bit words and check feasibility.
    # If we find one, we add it and continue (greedy).
    # This is a heuristic; the QR code is known optimal, so we expect no additions.
    # But if a nonlinear code of size 513 exists, this might find it.
    best = words[:]
    best_set = set(words)
    rng = random.Random(42)

    # Precompute bit masks for sampling
    all_ones = (1 << N) - 1
    candidates_tried = 0
    max_candidates = 500000  # Adjust based on time

    while time.time() < deadline and candidates_tried < max_candidates:
        # Sample a random 18-bit word
        cand = rng.randint(0, all_ones)
        candidates_tried += 1

        if cand in best_set:
            continue

        # Check distance to all existing words
        feasible = True
        for w in best:
            if (cand ^ w).bit_count() < MIN_DIST:
                feasible = False
                break

        if feasible:
            best.append(cand)
            best_set.add(cand)
            # Reset candidate limit to allow more search
            candidates_tried = 0
            max_candidates += 200000

    return best
