"""Optimized: compute minimal s for 5 consecutive nonzero QRs mod 1009, return with roots."""

P = 1009
M = 5

def construct():
    # Build QR lookup: qr[v] = a root r in [1, P-1] such that r*r % P == v, or 0 if v is not a nonzero QR
    qr_root = [0] * P
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and qr_root[r] == 0:
            qr_root[r] = x

    # Find minimal s in [1, P-M] such that s, s+1, ..., s+4 are all nonzero QRs
    # valid[v] = 1 if v is a nonzero QR, 0 otherwise
    valid = [0] * P
    for v in range(1, P):
        if qr_root[v] != 0:
            valid[v] = 1

    # Sliding window of size M
    window_sum = sum(valid[1:M+1])  # s=1: values 1..5
    if window_sum == M:
        roots = [qr_root[s] for s in range(1, M+1)]
        return {'start': 1, 'roots': roots}

    for s in range(2, P - M + 1):  # s=2..1004
        window_sum = window_sum - valid[s - 1] + valid[s + M - 1]
        if window_sum == M:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}

    # Fallback (should not reach here)
    raise RuntimeError('No run of 5 consecutive nonzero QRs found')
