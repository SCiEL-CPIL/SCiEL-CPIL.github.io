"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    MOD = 1009
    N = MOD - 1  # 1008

    # Precompute modular inverses for 1 to N
    # inv[x] is the unique y in 1..N such that x*y % MOD == 1
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 1
    best_length = 0

    # For each start s, find the maximum valid end e.
    # [s, e] is valid iff for all x in [s, e]: inv[x] < s or inv[x] > e.
    #
    # Correct O(n^2) approach: for each s, scan e from s to N,
    # maintaining the constraint that no x in [s,e] has inv[x] in [s,e].
    #
    # Key: we can precompute for each s the maximum valid end by checking
    # each candidate e and verifying the constraint incrementally.
    #
    # Actually, the simplest correct approach: for each s, extend e as far
    # as possible, but we must check ALL elements in [s, e], not just the new one.
    # However, we can be smarter: maintain a "blocking" minimum.
    #
    # For fixed s: as we increase e, the constraint is that for all x in [s, e],
    # inv[x] is not in [s, e]. Define:
    #   min_inv_in_window = min{ inv[x] : x in [s, e] }
    # The window is valid iff min_inv_in_window < s or min_inv_in_window > e.
    # But min_inv_in_window < s is the "good" case (inverse is outside to the left).
    # If min_inv_in_window >= s, we need min_inv_in_window > e.
    #
    # So: valid iff (for all x in [s,e], inv[x] < s) OR (min{inv[x]: x in [s,e]} > e)
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The condition is: for ALL x in [s,e], inv[x] NOT IN [s,e].
    # This means: for all x in [s,e], inv[x] < s OR inv[x] > e.
    #
    # Let m(s,e) = min{ inv[x] : x in [s,e] }.
    # If m(s,e) < s: then the minimum inverse is below s, but other inverses could still be in [s,e].
    # So we can't just use the minimum.
    #
    # Correct approach: for each s, the maximum valid e is determined by:
    # For each x >= s, if inv[x] >= s, then e must be < inv[x] (since if e >= inv[x] >= s,
    # and x <= e, then inv[x] is in [s,e], violating the condition).
    # So: e < min{ inv[x] : x in [s, N], inv[x] >= s }
    # (if the set is non-empty; otherwise e can be N)
    #
    # But we also need x <= e for the constraint to apply. So:
    # e < min{ inv[x] : x in [s, e], inv[x] >= s }
    # This is self-referential. Let's just do the direct O(n^2) scan.

    # Direct O(n^2): for each s, extend e and verify validity.
    # To verify [s,e] is valid: check all x in [s,e] that inv[x] is not in [s,e].
    # But we can do better: precompute for each s the answer.

    # For each s, compute max_valid_end[s]:
    # max_valid_end[s] = max e such that [s,e] is valid.
    #
    # Observation: if [s, e] is valid, then [s, e'] for e' < e is also valid.
    # And if [s, e] is invalid, then [s, e'] for e' > e may or may not be valid
    # (adding more elements can only add constraints, so it stays invalid).
    # Wait: if [s,e] is invalid because some x in [s,e] has inv[x] in [s,e],
    # then for e' > e, x is still in [s,e'] and inv[x] is still in [s,e'] (since inv[x] <= e < e').
    # So once invalid, always invalid for larger e. Good — monotonicity holds!
    #
    # So for each s, we can binary search or linearly scan for the max valid e.
    # But we need to check validity efficiently.
    #
    # Validity check for [s,e]: for all x in [s,e], inv[x] < s or inv[x] > e.
    # Equivalent: no x in [s,e] has s <= inv[x] <= e.
    #
    # Precomputation idea: for each position p (1..N), it is "blocked" for interval [s,e]
    # if s <= p <= e and s <= inv[p] <= e.
    #
    # For fixed s: an element p >= s blocks the interval if inv[p] >= s and inv[p] <= e.
    # So the maximum valid e for start s is:
    #   e_max(s) = min{ inv[p] - 1 : p in [s, N], inv[p] >= s }  (if such p exists)
    #   e_max(s) = N  (if no such p exists)
    #
    # But we also need e_max(s) >= s for a nonempty interval.
    # And we need to ensure that for all p in [s, e_max(s)], the constraint holds.
    # If e_max(s) = min{inv[p]-1 : p in [s,N], inv[p]>=s}, then for any p in [s, e_max(s)]:
    #   if inv[p] >= s, then inv[p] - 1 >= e_max(s) means inv[p] > e_max(s), so inv[p] > e_max(s).
    #   Wait, that's circular. Let me verify:
    #   e_max(s) = min{ inv[p] - 1 : p in [s,N], inv[p] >= s }
    #   For any p in [s, e_max(s)] with inv[p] >= s:
    #     inv[p] - 1 >= e_max(s) (by definition of min)
    #     So inv[p] >= e_max(s) + 1 > e_max(s)
    #     So inv[p] > e_max(s), meaning inv[p] is NOT in [s, e_max(s)]. ✓
    #   For any p in [s, e_max(s)] with inv[p] < s: inv[p] is NOT in [s, e_max(s)]. ✓
    # So [s, e_max(s)] is valid. And e_max(s) + 1 is invalid (by construction, there exists
    # some p in [s, N] with inv[p] = e_max(s) + 1 >= s, and if p <= e_max(s)+1, then
    # inv[p] = e_max(s)+1 is in [s, e_max(s)+1], making it invalid).
    # Wait, we need p <= e_max(s)+1 for the invalidity. The p that achieves the min might
    # be > e_max(s)+1. Hmm.
    #
    # Let me just do the direct O(n^2) with a clean implementation.

    # For each s, find max valid e by scanning.
    # Use the formula: e_max(s) = min{ inv[p] - 1 : p in [s, N], inv[p] >= s }, or N if empty.
    # Then verify [s, e_max(s)] is valid (should be by construction, but let's be safe).

    for s in range(1, N + 1):
        if time.time() > deadline:
            break

        # Compute e_max(s)
        min_inv = N + 1  # sentinel: larger than any valid value
        for p in range(s, N + 1):
            if inv[p] >= s:
                if inv[p] - 1 < min_inv:
                    min_inv = inv[p] - 1

        if min_inv == N + 1:
            e = N
        else:
            e = min_inv

        # e must be at least s for a nonempty interval
        if e < s:
            continue

        # Verify validity (safety check)
        valid = True
        for x in range(s, e + 1):
            if s <= inv[x] <= e:
                valid = False
                break

        if valid:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
# EVOLVE-BLOCK-END
