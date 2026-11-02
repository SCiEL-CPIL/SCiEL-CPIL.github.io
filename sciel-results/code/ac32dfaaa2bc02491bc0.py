"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _ball_indices(q, n, radius, center):
    """Return list of integer indices of all words within Hamming distance radius of center."""
    result = []
    # Generate all words within radius via multi-index enumeration
    for deltas in itertools.product(range(q), repeat=n):
        dist = sum(1 for i in range(n) if deltas[i] != 0)
        if dist > radius:
            continue
        idx = 0
        for i in range(n):
            val = (center[i] + deltas[i]) % q
            idx = idx * q + val
        result.append(idx)
    return result


def _ball_indices_fast(q, n, radius, center):
    """Faster ball enumeration using precomputed difference lists."""
    # For radius 1: center itself + n*(q-1) neighbors
    if radius == 1:
        base = 0
        for i in range(n):
            base = base * q + center[i]
        result = [base]
        for i in range(n):
            for d in range(q):
                if d == 0:
                    continue
                # Compute index with position i changed to (center[i]+d)%q
                idx = 0
                for j in range(n):
                    if j == i:
                        val = (center[i] + d) % q
                    else:
                        val = center[j]
                    idx = idx * q + val
                result.append(idx)
        return result
    # General case
    return _ball_indices(q, n, radius, center)


def _is_covered(idx, codewords, q, n, radius, balls):
    """Check if word at index idx is covered by at least one codeword."""
    for cw in codewords:
        dist = sum(1 for i in range(n) if cw[i] != (idx // (q ** (n - 1 - i))) % q)
        if dist <= radius:
            return True
    return False


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Initial code: direct sum of ternary tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute balls for all codewords
    balls = [_ball_indices_fast(q, n, radius, cw) for cw in codewords]

    # Phase 1: Greedy deletion - remove codewords whose ball is subset of union of others
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # For each codeword, check if its ball is fully covered by remaining codewords
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            ball_i = set(balls[i])
            # Check if every word in ball_i is covered by some other codeword
            uncovered = False
            for w in ball_i:
                found = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w in balls[j]:
                        found = True
                        break
                if not found:
                    uncovered = True
                    break
            if not uncovered:
                # Remove codeword i
                del codewords[i]
                del balls[i]
                changed = True
                break  # Restart loop since indices shifted

    # Phase 2: Local search - try swapping one codeword with another to improve coverage
    # Try replacing a codeword with a random one that covers more uncovered words
    for iteration in range(500):
        if time.time() > deadline:
            break

        # Compute coverage: which words are covered
        covered = [False] * (q ** n)
        for b in balls:
            for idx in b:
                covered[idx] = True

        # Find uncovered words
        uncovered_words = [i for i in range(q ** n) if not covered[i]]
        if not uncovered_words:
            break

        # Try to find a better codeword by random search
        best_gain = 0
        best_idx = -1
        best_new_cw = None
        best_new_ball = None

        # Sample random positions
        for _ in range(100):
            if time.time() > deadline:
                break
            new_cw = [random.randint(0, q - 1) for _ in range(n)]
            new_ball = _ball_indices_fast(q, n, radius, new_cw)

            # Count how many currently-uncovered words this would cover
            gain = sum(1 for w in new_ball if not covered[w])

            # Also consider: if we replace codeword i, we lose coverage of words only covered by i
            # For simplicity, just try pure additions first, then deletions
            if gain > best_gain:
                best_gain = gain
                best_new_cw = new_cw
                best_new_ball = new_ball

        if best_new_cw is not None and best_gain > 0:
            # Add the new codeword
            codewords.append(best_new_cw)
            balls.append(best_new_ball)

            # Now try to delete some codewords that are no longer needed
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ball_i = set(balls[i])
                uncovered = False
                for w in ball_i:
                    found = False
                    for j in range(len(codewords)):
                        if j == i:
                            continue
                        if w in balls[j]:
                            found = True
                            break
                    if not found:
                        uncovered = True
                        break
                if not uncovered:
                    del codewords[i]
                    del balls[i]

    # Final cleanup: remove any redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            ball_i = set(balls[i])
            uncovered = False
            for w in ball_i:
                found = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w in balls[j]:
                        found = True
                        break
                if not found:
                    uncovered = True
                    break
            if not uncovered:
                del codewords[i]
                del balls[i]
                changed = True
                break

    return codewords
# EVOLVE-BLOCK-END
