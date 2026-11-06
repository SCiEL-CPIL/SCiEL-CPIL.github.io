"""Find the minimum start s such that s, s+1, ..., s+4 are all quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s limit

    # Build a bitset of all non-zero quadratic residues mod p.
    # bit i is set iff i is a non-zero QR mod p.
    # Also store one (smallest) root for each residue.
    residue_bits = 0
    roots = [0] * p  # roots[r] = smallest x in [1,p-1] with x^2 % p == r, else 0

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)
            if roots[r] == 0:
                roots[r] = x

    # Check for 5 consecutive bits set using bit operations.
    # We need s, s+1, s+2, s+3, s+4 all set.
    # This is equivalent to: (residue_bits >> s) & 0x1F == 0x1F
    # We search s from 1 to 1004 (so s+4 <= 1008, no wraparound to 0).
    mask5 = 0x1F  # 5 bits all set

    for s in range(1, p - 4):
        if (residue_bits >> s) & mask5 == mask5:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}

    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
