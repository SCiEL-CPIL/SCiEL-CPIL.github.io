"""Rewritten program: maximize disjoint residue classes with distinct moduli 2..128."""

def construct():
    # Use moduli that are multiples of 11.
    # There are 11 distinct multiples of 11 in [2, 128]: 11, 22, ..., 121.
    # Assign residues 0, 1, ..., 10 respectively.
    # For any pair (m_i, m_j), gcd(m_i, m_j) is a multiple of 11.
    # Since r_i and r_j are distinct mod 11, they are distinct mod gcd(m_i, m_j).
    # Hence the residue classes are disjoint.
    # This yields 11 classes, which is optimal for this approach.
    moduli = [11 * k for k in range(1, 12)]  # 11, 22, ..., 121
    residues = list(range(11))               # 0, 1, ..., 10
    return [[r, m] for r, m in zip(residues, moduli)]
