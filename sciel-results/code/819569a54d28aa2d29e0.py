"""Find the longest interval in [1,1008] with no modular inverse pair (incl. diagonal)."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock
    MOD = 1009

    # Self-inverse residues: x^2 % MOD == 1 -> x in {1, MOD-1} = {1, 1008}.
    # These cannot appear in any valid interval (diagonal check x*x % MOD != 1 fails).
    # So restrict to [2, 1007].
    lo, hi = 2, 1007

    # Build inverse map for x in [2, 1007].
    inv = {}
    for x in range(lo, hi + 1):
        # Fermat: inv = x^(MOD-2) mod MOD
        r = pow(x, MOD - 2, MOD)
        inv[x] = r
        # r is in [1, 1008]; since x != 1,1008, r != x and r in [2,1007]
        assert r != x, (x, r)
        assert lo <= r <= hi, (x, r)

    # Partition [2,1007] into 2-cycles. Assign each cycle an id.
    cycle_id = {}
    seen = set()
    next_id = 0
    for x in range(lo, hi + 1):
        if x in seen:
            continue
        y = inv[x]
        if x < y:
            a, b = x, y
        else:
            a, b = y, x
        cycle_id[a] = next_id
        cycle_id[b] = next_id
        seen.add(a)
        seen.add(b)
        next_id += 1
    num_cycles = next_id

    # Sliding window over positions [lo, hi] (contiguous integers).
    # A window is valid iff no cycle has both members inside it.
    # count[c] = number of members of cycle c currently in window.
    count = [0] * num_cycles
    best_start = lo
    best_len = 0
    left = lo
    for right in range(lo, hi + 1):
        c = cycle_id[right]
        count[c] += 1
        # Shrink from left while some cycle has count 2.
        while count[c] == 2:
            cl = cycle_id[left]
            count[cl] -= 1
            left += 1
            if left > right:
                break
            # re-evaluate: the offending cycle may still be 2
            # We must loop until no cycle has count 2.
            # Simpler: keep shrinking while any count==2. Track via c after removal.
            # Recompute offending: we removed left; the cycle that was 2 may now be 1.
            # But another cycle could be 2. Use a while with a flag.
            # Re-do: break inner and rely on outer while condition rechecked below.
        # The above while only checks count[c]==2 for the just-added cycle.
        # We need to ensure NO cycle has count 2. Maintain a variable 'bad' = number of cycles with count 2.
        # Redo properly below.
        # (placeholder; replaced by correct version)
        if time.time() > deadline:
            break

    # Correct sliding window with a 'bad' counter.
    count = [0] * num_cycles
    bad = 0  # number of cycles with count == 2
    best_start = lo
    best_len = 0
    left = lo
    for right in range(lo, hi + 1):
        c = cycle_id[right]
        count[c] += 1
        if count[c] == 2:
            bad += 1
        while bad > 0:
            cl = cycle_id[left]
            count[cl] -= 1
            if count[cl] == 1:
                bad -= 1
            left += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        cur_len = right - left + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = left

    # Fallback if nothing found (shouldn't happen).
    if best_len <= 0:
        return [2, 1]

    # Verify validity of the chosen interval (defensive, cheap).
    start = best_start
    length = best_len
    present = set(range(start, start + length))
    for x in present:
        if x * x % MOD == 1:
            # self-inverse present -> invalid, shouldn't happen
            return [2, 1]
        r = inv.get(x)
        if r is not None and r in present:
            # full cycle present -> invalid, shouldn't happen
            return [2, 1]

    return [start, length]
# EVOLVE-BLOCK-END
