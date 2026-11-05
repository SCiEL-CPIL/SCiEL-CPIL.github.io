import time

P = 1009
M = 5
DEADLINE_MARGIN = 2.0  # seconds to stop before the 90s wall-clock limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    # Build is_qr and roots in one pass
    is_qr = [False] * P
    roots = [0] * P

    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            if roots[r] == 0:
                roots[r] = x

    # Search for minimum s in [1, P-M] such that s..s+4 are all nonzero QRs
    for s in range(1, P - M + 1):
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            result_roots = [roots[s], roots[s + 1], roots[s + 2], roots[s + 3], roots[s + 4]]

            # Validate each root
            for i in range(M):
                r = result_roots[i]
                if (r * r) % P != s + i:
                    # Fallback: find a valid root
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            result_roots[i] = x
                            break

            if time.time() > deadline:
                # Still return the valid result we found
                pass
            return {'start': s, 'roots': result_roots}

    # Fallback: descending search (should not be needed for P=1009)
    for s in range(P - M, 0, -1):
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            result_roots = [roots[s], roots[s + 1], roots[s + 2], roots[s + 3], roots[s + 4]]
            for i in range(M):
                r = result_roots[i]
                if (r * r) % P != s + i:
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            result_roots[i] = x
                            break
            return {'start': s, 'roots': result_roots}

    raise RuntimeError("No run found within deadline or search space")
