"""Program for A_2(17,6) code construction via randomized local search from linear QR code."""
import time
import random
import numpy as np

N = 17
MIN_DIST = 6
DEADLINE_FRACTION = 0.85  # safety margin


def _hamming_dist(a, b):
    return (a ^ b).bit_count()


def _build_linear_qr_code():
    """
    Construct the 256-word [17,8,6] code (shortened extended QR code).
    We build the [18,9,6] extended QR code and then shorten by one coordinate.
    
    Actually, let's use a more direct approach: generate a [17,8,6] linear code.
    The extended quadratic residue code of length 18 has dimension 9 and distance 6.
    Shortening by one coordinate gives [17,8,6] with 256 words.
    
    A simpler approach: use the known generator matrix for a [17,8,6] code.
    We can construct it from the quadratic residues mod 17.
    """
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # The [17,8,6] code can be constructed as follows:
    # Generator matrix G is 8x17.
    # Use the standard construction: rows are the characteristic vectors of 
    # cyclic shifts of the quadratic residue set, plus an all-ones row for shortening.
    
    # Actually, let me use a direct construction.
    # The [18,9,6] extended QR code: take the [17,8,6] QR code and add a parity bit.
    # The [17,8,6] QR code generator: 8x17 matrix where row i has 1s at positions
    # corresponding to quadratic residues shifted by i (mod 17).
    
    # Let me just build a valid [17,8,6] code directly.
    # Quadratic residues mod 17: {1,2,4,8,9,13,15,16}
    # Position 0 is the "infinity" point.
    
    # Generator matrix for [17,8,6] code:
    # Row 0: all ones (the shortening coordinate)
    # Rows 1-8: cyclic shifts of the QR set
    
    # Actually, the standard [17,8,6] QR code has generator matrix:
    # G = [I_8 | P] where P is an 8x9 matrix.
    # Let me use a known construction.
    
    # Simpler: use the fact that the [17,8,6] code is the dual of the [17,9,5] code.
    # Or just construct it directly from the QR set.
    
    # The [17,8,6] quadratic residue code:
    # Let Q = {0, 1, 2, 4, 8, 9, 13, 15, 16} (including 0 as infinity)
    # Wait, there are 9 quadratic residues including 0 mod 17.
    # The [17,8,6] code: generator matrix has rows that are characteristic vectors
    # of the 8 non-zero quadratic residues, with appropriate shifts.
    
    # Let me just use a concrete generator matrix.
    # The [17,8,6] QR code generator matrix (one valid form):
    
    # Actually, the simplest correct approach: 
    # Build the [18,9,6] extended QR code, then shorten.
    # The [18,9,6] code: 9-dimensional, min distance 6.
    # Shortening: fix one coordinate to 0, get [17,8,6] with 256 words.
    
    # Let me construct the [18,9,6] code:
    # Generator matrix G_ext is 9x18.
    # The [17,8,6] code is obtained by taking all codewords of [18,9,6] with last bit = 0.
    
    # For the [18,9,6] extended QR code mod 17:
    # The generator matrix can be built from the quadratic residues.
    
    # Let me use a different, verified approach.
    # I'll construct a [17,8,6] code using a known generator matrix.
    
    # Here's a valid [17,8,6] generator matrix (8x17):
    # Row i (0-indexed) has 1s at positions: {i} union {(i + q) mod 17 : q in QR_nonzero}
    # where QR_nonzero = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Wait, that gives each row weight 9, and the code would have 2^8 = 256 words.
    # Let me verify the minimum distance is 6.
    
    qr_nonzero = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Generator matrix: 8 rows, each a 17-bit integer
    gen_rows = []
    for i in range(8):
        row = 1 << i  # position i
        for q in qr_nonzero:
            pos = (i + q) % 17
            row |= (1 << pos)
        gen_rows.append(row)
    
    # Generate all 256 codewords
    words = []
    for mask in range(256):
        word = 0
        for j in range(8):
            if (mask >> j) & 1:
                word ^= gen_rows[j]
        words.append(word)
    
    return words


def _validate(words):
    """Check all pairwise distances are at least MIN_DIST."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if _hamming_dist(words[i], words[j]) < MIN_DIST:
                return False
    return True


def _try_add_word(words, candidate):
    """Check if candidate can be added to words."""
    for w in words:
        if _hamming_dist(w, candidate) < MIN_DIST:
            return False
    return True


def _repair(words, candidate, max_flips=3):
    """
    Try to repair the code by flipping bits in existing codewords to make room for candidate.
    Returns (new_words, success) where new_words includes candidate if successful.
    """
    n = len(words)
    
    # Find conflicts: codewords too close to candidate
    conflicts = []
    for i, w in enumerate(words):
        d = _hamming_dist(w, candidate)
        if d < MIN_DIST:
            conflicts.append((i, w, d))
    
    if not conflicts:
        return words + [candidate], True
    
    if len(conflicts) > 2:
        return words, False
    
    # Try to fix conflicts by flipping bits in the conflicting codewords
    # For each conflict, find a bit flip that increases distance to candidate
    # while maintaining distance to all other codewords
    
    for idx, (ci, cw, cd) in enumerate(conflicts):
        # Need to increase distance from cd to at least MIN_DIST
        # Flip a bit in cw that is 0 where candidate is 1, or 1 where candidate is 0
        # to increase the Hamming distance
        
        # Try all single-bit flips
        for bit in range(N):
            new_cw = cw ^ (1 << bit)
            # Check if this increases distance to candidate
            new_cd = _hamming_dist(new_cw, candidate)
            if new_cd < MIN_DIST:
                continue
            # Check distance to all other codewords
            valid = True
            for j, w in enumerate(words):
                if j == ci:
                    continue
                if _hamming_dist(new_cw, w) < MIN_DIST:
                    valid = False
                    break
            if valid:
                # Also check against other conflicts
                words[ci] = new_cw
                # If there are multiple conflicts, we'd need to handle them together
                # For simplicity, only handle single conflicts
                if len(conflicts) == 1:
                    return words + [candidate], True
    
    return words, False


def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with the 256-word linear code
    words = _build_linear_qr_code()
    best_words = list(words)
    best_count = len(words)
    
    if not _validate(best_words):
        # Fallback to the 32-word Reed-Muller code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        best_words = list(words)
        best_count = len(words)
    
    random.seed(42)
    
    # Simulated annealing / local search
    # Try to add random words, with repair
    while time.time() < deadline:
        # Generate a random candidate
        candidate = random.getrandbits(N)
        
        # Try to add directly
        if _try_add_word(words, candidate):
            words.append(candidate)
            if len(words) > best_count:
                best_words = list(words)
                best_count = len(words)
            continue
        
        # Try repair
        new_words, success = _repair(words, candidate, max_flips=2)
        if success:
            words = new_words
            if len(words) > best_count:
                best_words = list(words)
                best_count = len(words)
    
    # Final validation
    if not _validate(best_words):
        # Return a valid subset
        valid = []
        for w in best_words:
            ok = True
            for v in valid:
                if _hamming_dist(w, v) < MIN_DIST:
                    ok = False
                    break
            if ok:
                valid.append(w)
        best_words = valid
    
    if not best_words:
        # Ultimate fallback
        best_words = [0]
    
    return best_words
