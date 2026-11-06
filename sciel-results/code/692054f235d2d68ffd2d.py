"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The maximum number of disjoint residue classes with distinct moduli in [2, 128]
    # is determined by finding a prime p such that we can pick p distinct multiples of p
    # in the range, and assign distinct residues modulo p.
    # For p=11, there are floor(128/11) = 11 multiples: 11, 22, ..., 121.
    # We assign residues 0 to 10.
    # Since gcd(11*k, 11*l) is a multiple of 11, and residues are distinct mod 11,
    # the classes are disjoint.
    p = 11
    moduli = [p * k for k in range(1, 12) if p * k <= 128]
    # Ensure we have exactly p moduli to maximize the count
    # If we had more, we couldn't assign distinct residues mod p.
    # Here len(moduli) is 11.
    residues = list(range(p))
    
    result = []
    for i, m in enumerate(moduli):
        r = residues[i % p]
        result.append([r, m])
        
    return result
# EVOLVE-BLOCK-END
