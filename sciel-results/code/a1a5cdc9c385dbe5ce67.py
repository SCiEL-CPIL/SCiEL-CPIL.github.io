import time


def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s kill

    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Identify self-inverse elements (x*x % p == 1)
    is_self_inv = [False] * (p + 1)
    for x in range(1, p):
        if (x * x) % p == 1:
            is_self_inv[x] = True

    # Partner: for non-self-inverse x, partner[x] = inv[x]
    partner = [0] * (p + 1)
    for x in range(1, p):
        if not is_self_inv[x]:
            partner[x] = inv[x]

    # Two-pointer sliding window:
    # Maintain window [L, R] that is valid (no self-inverse, no inverse pair both present)
    # For each R, expand; if conflict, shrink L until valid.
    # A conflict exists when:
    #   - R is self-inverse, OR
    #   - partner[R] is already in the window
    # Since each residue appears at most once in [1,1008], we just track presence.

    best_start = 2
    best_length = 1

    L = 1
    present = [False] * (p + 1)
    conflict_count = 0

    for R in range(1, p):
        # Add R to window
        present[R] = True
        if is_self_inv[R]:
            conflict_count += 1
        else:
            pr = partner[R]
            if pr != 0 and present[pr]:
                conflict_count += 1

        # Shrink L until window is valid
        while conflict_count > 0 and L <= R:
            if is_self_inv[L]:
                conflict_count -= 1
            else:
                pr = partner[L]
                if pr != 0 and present[pr]:
                    conflict_count -= 1
            present[L] = False
            L += 1

        # Now [L, R] is valid
        current_len = R - L + 1
        if current_len > best_length:
            best_length = current_len
            best_start = L

        # Time guard
        if time.time() > deadline:
            break

    # Final verification of the best answer
    if best_length >= 1 and best_start >= 1 and best_start + best_length - 1 <= 1008:
        # Quick check: ensure no self-inverse and no inverse pair in [best_start, best_start+best_length-1]
        end = best_start + best_length - 1
        valid = True
        for x in range(best_start, end + 1):
            if is_self_inv[x]:
                valid = False
                break
            ix = inv[x]
            if best_start <= ix <= end:
                valid = False
                break
        if valid:
            return [best_start, best_length]

    # Fallback: return a trivially valid 1-element interval not containing self-inverse
    for x in range(2, p):
        if not is_self_inv[x]:
            return [x, 1]

    return [2, 1]
