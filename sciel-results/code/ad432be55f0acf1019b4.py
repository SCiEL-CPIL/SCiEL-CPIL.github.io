"""Minimal run-start s for 5 consecutive non-zero quadratic residues mod 1009.

r(k,m,p) with k=2 (quadratic), m=5 (run length), p=1009.
Return {'start':s,'roots':[r0..r4]} where r_i^2 mod 1009 == s+i, s in [1,1004].
Minimize s; zero residues and wraparound excluded.
"""
import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0


def _elapsed():
    return time.time() - _START


def construct():
    # Map residue -> one root r in [1, P-1] with r^2 % P == residue (skip residue 0).
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Build a bitset: bit v is 1 iff v (1..1008) is a non-zero quadratic residue.
    bitmask = 0
    for v in range(1, _P):
        if root[v] != 0:
            bitmask |= (1 << v)

    # To find 5 consecutive residues starting at s, we need bits s, s+1, s+2, s+3, s+4 all set.
    # This is equivalent to: (bitmask >> s) & 0b11111 == 0b11111
    # Use a single shifted-mask AND to find all valid starts at once, then locate the least set bit.
    # Valid s range: 1 <= s <= 1004 (so s+4 <= 1008, no wraparound).
    
    # Create a mask that has 1s in positions 1 through 1004 (inclusive).
    # We want to find positions s in [1, 1004] where (bitmask >> s) & 0b11111 == 0b11111.
    # Equivalently, for each s in [1, 1004], check if all 5 bits are set.
    # 
    # Alternative efficient approach: 
    # Let mask5 = 0b11111. For each valid s, (bitmask & (mask5 << s)) == (mask5 << s).
    # We can compute: candidates = bitmask
    # For each offset i in 0..4, we need bit (s+i) set. 
    # So: all_set = bitmask
    # For i in 1..4: all_set &= (bitmask >> i)  -- no, that shifts the whole thing.
    # 
    # Better: compute a "consecutive mask" where bit s is set iff bits s..s+4 are all set in bitmask.
    # consecutive = bitmask
    # for i in range(1, 5):
    #     consecutive &= (bitmask >> i)  -- this doesn't work directly because we need to align.
    #
    # Correct approach:
    # consecutive = bitmask
    # for i in range(1, 5):
    #     consecutive &= (bitmask >> i)
    # Wait, that's not right either. Let me think again.
    #
    # If we want bit s in the result to be 1 iff bits s, s+1, s+2, s+3, s+4 are all 1 in bitmask:
    # result = bitmask
    # for i in range(1, 5):
    #     result &= (bitmask >> i)
    # No, that's wrong. (bitmask >> i) has bit s equal to bitmask bit s+i. So:
    # result = bitmask
    # for i in range(1, 5):
    #     result &= (bitmask >> i)
    # This gives: result bit s = bitmask[s] & bitmask[s+1] & bitmask[s+2] & bitmask[s+3] & bitmask[s+4]
    # Yes! That's correct.
    
    consecutive = bitmask
    for i in range(1, 5):
        consecutive &= (bitmask >> i)
    
    # Now bit s in consecutive is 1 iff s, s+1, s+2, s+3, s+4 are all quadratic residues.
    # We need the least s in [1, 1004] such that bit s is set.
    # Mask out positions 0 and positions > 1004.
    # Positions 1..1004 inclusive.
    valid_mask = 0
    for s in range(1, _P - 4):  # 1 to 1004
        valid_mask |= (1 << s)
    
    candidates = consecutive & valid_mask
    
    if candidates == 0:
        # Fallback: scan linearly (should not happen for p=1009)
        max_s = _P - 5  # 1004
        for s in range(1, max_s + 1):
            if (bitmask >> s) & 0b11111 == 0b11111:
                roots = [root[s + i] for i in range(5)]
                return {'start': s, 'roots': roots}
            if _elapsed() > _DEADLINE - _MARGIN:
                break
        raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
    
    # Find the least set bit in candidates.
    # Least set bit: candidates & (-candidates), then find the index.
    least_bit = candidates & (-candidates)
    s = least_bit.bit_length() - 1
    
    # Verify s is in valid range
    if s < 1 or s > _P - 5:
        # Should not happen, but fallback
        max_s = _P - 5
        for s in range(1, max_s + 1):
            if (bitmask >> s) & 0b11111 == 0b11111:
                roots = [root[s + i] for i in range(5)]
                return {'start': s, 'roots': roots}
        raise RuntimeError('No valid 5-run found')
    
    roots = [root[s + i] for i in range(5)]
    return {'start': s, 'roots': roots}
