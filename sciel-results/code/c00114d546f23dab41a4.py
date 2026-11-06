import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    p = 1009
    
    # Build a dictionary mapping each non-zero quadratic residue to a root
    # x such that x^2 mod p == residue. We store the smallest such x (1..p-1).
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Find the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4
    # are all non-zero quadratic residues (i.e., all in roots).
    # Note: s+4 <= 1008 when s <= 1004, so no wraparound needed.
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            break
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    # Fallback: should not happen, but if we hit the deadline without finding a result,
    # we need to return something valid. Since the problem guarantees a solution exists
    # and s=1..1004 is a small range, this is unreachable in practice.
    # But to be safe, we could do a quick re-check or raise.
    raise RuntimeError('No baseline run found')
