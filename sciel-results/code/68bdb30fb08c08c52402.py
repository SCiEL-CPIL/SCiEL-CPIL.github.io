"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    p = 1009

    # Build a bitset where bit i is set iff i is a non-zero quadratic residue mod p
    # We represent positions 1..p-1 as bits 0..p-2 (i.e., bit (x-1) corresponds to value x)
    # Actually, let's use bit x to represent value x, for x in 1..p-1
    # So we need a bitset of length p (bits 0..p-1), where bit x is 1 if x is a QR mod p
    # and x != 0.

    is_qr = [False] * p  # index 0..p-1, is_qr[x] is True if x is a non-zero QR mod p
    root_map = {}  # maps QR value to a root in 1..p-1

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if r not in root_map:
                root_map[r] = x

    # Build an integer bitset: bit i (0-indexed) corresponds to value i+1? 
    # No, let's make it simpler: bit i corresponds to value i, for i in 1..p-1.
    # So we need a big integer where bit i is set if is_qr[i] is True, for i in 1..p-1.
    # bit 0 corresponds to value 0 (always 0, excluded)
    # bit 1 corresponds to value 1
    # ...
    # bit p-1 corresponds to value p-1

    mask = 0
    for i in range(1, p):
        if is_qr[i]:
            mask |= (1 << i)

    # Now, to find runs of 5 consecutive QRs starting at s, we need:
    # bits s, s+1, s+2, s+3, s+4 all set in mask
    # This is equivalent to: (mask >> s) & 0b11111 == 0b11111
    # i.e., mask & (0b11111 << s) == (0b11111 << s)

    # We want the minimum s in [1, p-5] such that this holds.
    # p-5 = 1004, so s in [1, 1004].

    five_mask = 0b11111  # 31

    # Shift the mask right by s and check if the lowest 5 bits are all 1
    # Equivalently: (mask >> s) & 31 == 31

    # We can iterate s from 1 upward and check.
    # But we can also do a smarter approach: 
    # Let shifted = mask >> s. We need (shifted & 31) == 31.
    # This means bits s through s+4 of mask are all 1.

    # Let's just iterate s from 1 to p-5 (1004) and check.
    # This is at most 1004 iterations, each O(1) with big integer ops.

    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            # Safety fallback: return a known valid solution if we have one
            # For p=1009, we should find one quickly
            raise RuntimeError("Deadline exceeded")
        
        if (mask >> s) & five_mask == five_mask:
            # All five values s, s+1, s+2, s+3, s+4 are QRs
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No solution found")
# EVOLVE-BLOCK-END
