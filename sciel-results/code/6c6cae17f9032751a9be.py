"""CSP forward-checking solver for increasing sequences with unique consecutive-block sums."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # First, get a baseline via fast greedy
    greedy_seq = _forward_greedy()
    best_seq = greedy_seq
    best_len = len(greedy_seq)

    # Try to improve via randomized greedy with different heuristics
    # We'll try multiple randomized restarts
    for attempt in range(200):
        if time.time() > deadline:
            break

        # Different strategies per attempt
        if attempt % 3 == 0:
            # Standard greedy but with slight randomization in tie-breaking
            seq = _randomized_greedy(1, 256, randomize_factor=0.1)
        elif attempt % 3 == 1:
            # Greedy with lookahead: prefer numbers that create larger gaps
            seq = _greedy_with_lookahead(1, 256)
        else:
            # Start from a random offset
            seq = _randomized_greedy(1 + (attempt % 5), 256, randomize_factor=0.2)

        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    # Try insertion-based improvement: can we insert one more element?
    # This is O(n * 256 * verification_cost)
    for _ in range(100):
        if time.time() > deadline:
            break

        current_len = len(best_seq)
        if current_len >= 25:
            break

        # Try inserting at each position
        improved = False
        for pos in range(current_len + 1):
            if time.time() > deadline:
                break

            # Determine valid range for new element at this position
            low = best_seq[pos - 1] + 1 if pos > 0 else 1
            high = best_seq[pos] - 1 if pos < current_len else 256

            if low > high:
                continue

            # Try a few candidate values in this range
            candidates = list(range(low, high + 1))
            # If too many, sample
            if len(candidates) > 20:
                import random
                candidates = random.sample(candidates, 20)

            for x in candidates:
                if time.time() > deadline:
                    break

                new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                if _is_valid_fast(new_seq):
                    best_seq = new_seq
                    best_len += 1
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return best_seq


def _forward_greedy():
    """Standard greedy: pick smallest valid number."""
    values = []
    sums = set()

    for x in range(1, 257):
        # Calculate new block sums ending at x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)

        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)

    return values


def _randomized_greedy(start, end, randomize_factor=0.1):
    """Greedy with slight randomization: sometimes pick a non-minimal valid candidate."""
    import random
    values = []
    sums = set()

    for x in range(start, end + 1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)

        if not sums.intersection(tails):
            # With probability randomize_factor, skip this and try a larger valid one
            if random.random() < randomize_factor and x < end:
                # Look for next valid
                found = False
                for y in range(x + 1, min(x + 10, end + 1)):
                    t2 = [y]
                    tot2 = y
                    for a in reversed(values):
                        tot2 += a
                        t2.append(tot2)
                    if not sums.intersection(t2):
                        values.append(y)
                        sums.update(t2)
                        found = True
                        break
                if not found:
                    # Fall back to x
                    values.append(x)
                    sums.update(tails)
            else:
                values.append(x)
                sums.update(tails)

    return values


def _greedy_with_lookahead(start, end):
    """Greedy that prefers candidates creating larger minimum gap to existing sums."""
    import random
    values = []
    sums = set()

    for x in range(start, end + 1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)

        if not sums.intersection(tails):
            # Check if there's a slightly larger candidate with better "spread"
            best_choice = x
            best_score = -1

            for y in range(x, min(x + 5, end + 1)):
                t2 = [y]
                tot2 = y
                for a in reversed(values):
                    tot2 += a
                    t2.append(tot2)
                if not sums.intersection(t2):
                    # Score: how far the new sums are from existing sums
                    min_dist = float('inf')
                    for s in t2:
                        for es in sums:
                            d = abs(s - es)
                            if d < min_dist:
                                min_dist = d
                    score = min_dist if min_dist != float('inf') else 1000
                    if score > best_score:
                        best_score = score
                        best_choice = y

            values.append(best_choice)
            # Update sums with the chosen value
            t2 = [best_choice]
            tot2 = best_choice
            for a in reversed(values[:-1]):
                tot2 += a
                t2.append(tot2)
            sums.update(t2)

    return values


def _is_valid_fast(seq):
    """Fast validity check for a strictly increasing sequence."""
    if not seq:
        return True

    # Check strictly increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False

    # Check unique block sums
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_valid(seq):
    """Check if sequence is strictly increasing and all consecutive block sums are unique."""
    return _is_valid_fast(seq)
# EVOLVE-BLOCK-END
