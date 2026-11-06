"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _subset_sums_distinct(vals):
    """Check if all 2^n subset sums are distinct for sorted vals."""
    n = len(vals)
    # Use a set of sums
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        # Check for collisions before adding
        for ns in new_sums:
            if ns in sums:
                return False
        sums.update(new_sums)
    return True

def _try_greedy(max_val):
    """
    Greedy: pick smallest a_i such that all subset sums remain distinct.
    Returns list of 12 integers or None if impossible within max_val.
    """
    vals = []
    sums = {0}
    for i in range(12):
        # Try candidates starting from max(vals) + 1 (or 1 if empty)
        start = vals[-1] + 1 if vals else 1
        # Upper bound: must be <= max_val
        if start > max_val:
            return None
        for cand in range(start, max_val + 1):
            # Check if adding cand creates any duplicate subset sum
            # A new sum (s + cand) collides with an existing sum t iff
            # t - cand = s' where s' is in sums and t is in sums
            # More efficiently: check if (sums + cand) intersects sums
            # i.e., exists s1, s2 in sums with s1 + cand = s2 => s2 - s1 = cand
            # So cand must not be in {s2 - s1 : s1, s2 in sums, s1 < s2}
            # But that's expensive. Instead just do the set check.
            collides = False
            for s in sums:
                if s + cand in sums:
                    collides = True
                    break
            if not collides:
                vals.append(cand)
                for s in list(sums):
                    sums.add(s + cand)
                break
        else:
            return None
    return vals

def _optimize_greedy(start_max, end_max, deadline):
    """
    Try greedy for various max_val bounds to find the minimal max_val that works.
    Actually, just run greedy with a high enough bound and see what max it uses.
    """
    best = None
    # First, try with a reasonable bound
    for bound in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        if time.time() > deadline:
            break
        vals = _try_greedy(bound)
        if vals is not None:
            candidate_max = max(vals)
            if best is None or candidate_max < best:
                best = candidate_max
                best_vals = vals
                # Don't break; a smaller bound might give smaller max? 
                # No, greedy always picks smallest possible, so the result is the same regardless of bound (as long as bound is large enough)
                # Actually, if bound is too small, it might fail. If it succeeds, the max is determined by the greedy choices.
                # The greedy algorithm is deterministic: it always picks the smallest valid next element.
                # So if it succeeds with any bound, it gives the same sequence.
                # Thus we can just try once with a large bound.
                break
    if best_vals is not None:
        return best_vals
    return None

def construct():
    # The minimal maximum element for n distinct positive integers with distinct subset sums is 2^(n-1).
    # For n=12, this is 2048. The set {1, 2, 4, ..., 2048} achieves this.
    # Any other set must have a maximum element >= 2048.
    # Thus, returning the powers of 2 is optimal.
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
