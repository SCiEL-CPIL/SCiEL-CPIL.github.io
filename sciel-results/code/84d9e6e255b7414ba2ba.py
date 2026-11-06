"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Strategy: pick a prime p, and use p distinct multiples of p in [2, 128].
    # Assign residues 0..p-1. Since all moduli are multiples of p, any two moduli
    # m_i, m_j have gcd >= p. The CRT criterion for disjointness of classes
    # (r_i mod m_i) and (r_j mod m_j) is that r_i != r_j mod gcd(m_i, m_j).
    # Since gcd(m_i, m_j) is a multiple of p, it suffices that r_i != r_j mod p.
    # With distinct residues 0..p-1, this holds.
    # Maximize min(floor(128/p), p) over primes p.
    # p=11: floor(128/11)=11, min=11.
    # p=13: floor(128/13)=9, min=9.
    # p=7: floor(128/7)=18, min=7.
    # So p=11 gives 11 classes.

    p = 11
    moduli = [p * k for k in range(1, p + 1)]  # 11, 22, ..., 121
    residues = list(range(p))

    result = [[residues[i], moduli[i]] for i in range(p)]
    return result
# EVOLVE-BLOCK-END
