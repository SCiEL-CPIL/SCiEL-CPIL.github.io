import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 80.0

    MOD = 1009
    N = MOD - 1  # 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Build list of "conflict boundaries":
    # For each element x in [1, 1008], its inverse inv[x] creates a constraint.
    # We need to find the longest contiguous interval where no x,y in the interval
    # satisfy x*y % MOD == 1.
    #
    # Key insight: For each position r, define L_r = the minimum left endpoint such
    # that [L_r, r] is valid. We track this with a two-pointer approach.
    #
    # Structural change: Instead of a simple two-pointer, we precompute all
    # "forbidden pairs" and use a sweep-line with a priority structure to
    # find the maximum gap more explicitly.
    #
    # For each x in [1, N], the pair (x, inv[x]) is forbidden if x != inv[x].
    # If x == inv[x], then x cannot appear in any valid interval.
    #
    # We collect all constraints as intervals that cannot both be in a valid window.
    # For x < inv[x]: the pair forbids any window containing both x and inv[x].
    # For x > inv[x]: same pair, already counted.
    # For x == inv[x]: x is a "poison" that splits the space.
    #
    # New approach: Create a boolean array "blocked" where blocked[i] = True if
    # i is a self-inverse (cannot be in any valid interval). Then, for non-poison
    # positions, we need to ensure no two positions in the window are inverses.
    #
    # We'll use a different formulation: for each right endpoint r (skipping poisons),
    # maintain the maximum left constraint imposed by all elements in [1..r].

    # Mark self-inverse elements
    poison = [False] * (MOD + 1)
    for x in range(1, MOD):
        if inv[x] == x:
            poison[x] = True

    # For each element x (not poison), its inverse inv[x] creates a mutual constraint.
    # If we include x in the window, we cannot include inv[x], and vice versa.
    # We process right endpoints in order, maintaining the tightest left bound.

    best_start = 1
    best_len = 1  # At least one valid element exists

    left_bound = 1  # Current left boundary of the valid window ending at 'right'

    for right in range(1, MOD):
        if time.time() > deadline:
            break

        if poison[right]:
            # right cannot be in any valid interval, so it acts as a wall
            left_bound = right + 1
            continue

        ix = inv[right]

        # If ix is also not poison and ix >= left_bound, we must move left past ix
        if ix >= left_bound:
            left_bound = ix + 1

        # If ix is poison, it's already excluded, no additional constraint needed
        # (it can't be in the window anyway)

        curr_len = right - left_bound + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = left_bound

    # Verify the result
    end = best_start + best_len - 1
    valid = True
    if best_start < 1 or end > N:
        valid = False
    else:
        for i in range(best_start, end + 1):
            if poison[i]:
                valid = False
                break
            for j in range(i, end + 1):
                if (i * j) % MOD == 1:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback: find any valid single element
        for s in range(1, MOD):
            if not poison[s]:
                best_start = s
                best_len = 1
                break

    return [best_start, best_len]

# EVOLVE-BLOCK-END
