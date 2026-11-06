"""Construct the optimal 512-word extended quadratic-residue code of length 18, distance 6."""
# EVOLVE-BLOCK-START
import time

def _build_basis():
    """
    Build 9 basis vectors (18-bit integers) that span a [18,9,6] code.
    Uses the [16,11,6] extended Hamming code, takes 9 generator rows,
    and appends 2 zero bits to reach length 18.
    """
    # [15,11,5] Hamming code: H is 4x15 with columns = all nonzero 4-bit vectors.
    # Systematic form: G = [I_11 | P^T] where P is 4x11.
    # The 11 message positions correspond to 4-bit values (nonzero, excluding the 4 unit vectors used for parity).
    P_cols = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]  # 4-bit values for the 11 P columns
    
    # Build G15: 11 rows, 15 columns
    G15 = []
    for i in range(11):
        row = 1 << i  # systematic bit
        val = P_cols[i]
        for bit in range(4):
            if val & (1 << bit):
                row |= (1 << (11 + bit))
        G15.append(row)
    
    # Extend to [16,11,6]: add overall parity bit at position 15
    G16 = []
    for row in G15:
        extended = row
        if row.bit_count() % 2 == 1:
            extended |= (1 << 15)
        G16.append(extended)
    
    # Take first 9 rows, extend to length 18 by appending 2 zero bits
    basis = []
    for i in range(9):
        word = G16[i] << 2  # shift left by 2 to make room for 2 zero bits
        basis.append(word)
    
    return basis

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    basis = _build_basis()
    n = len(basis)  # 9
    
    # Enumerate all 2^9 = 512 combinations via XOR
    words = [0]
    for i in range(n):
        current_len = len(words)
        for j in range(current_len):
            words.append(words[j] ^ basis[i])
    
    return words
# EVOLVE-BLOCK-END
