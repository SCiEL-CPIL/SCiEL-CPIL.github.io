"""Rewritten program: structural change to simulated-annealing + dense AP evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _build_ap_indices(n):
    """Return lists (starts, steps, lengths) for every AP in [0, n-1]."""
    starts = []
    steps = []
    lengths = []
    for step in range(1, n + 1):
        for start in range(n):
            length = (n - 1 - start) // step + 1
            if length >= 1:
                starts.append(start)
                steps.append(step)
                lengths.append(length)
    return np.array(starts, dtype=np.int32), np.array(steps, dtype=np.int32), np.array(lengths, dtype=np.int32)

def _eval_all(signs, starts, steps, lengths):
    """Compute max |sum| over all APs. signs is np.int8 array of length n."""
    # For each AP, sum signs[start::step] up to length
    # Vectorize over starts for a given step? We'll do a loop over steps but vectorize starts.
    n = len(signs)
    max_val = 0
    # Group by step
    unique_steps = np.unique(steps)
    for s in unique_steps:
        mask = steps == s
        ss = starts[mask]
        ls = lengths[mask]
        # For each start in ss, sum signs[start:start+ls*s:s]
        # We can do this with a loop over ss, but that's slow.
        # Alternative: for a fixed step, the APs are disjoint in terms of starting residue class?
        # Actually for fixed step s, AP starting at r (r < s) has length floor((n-1-r)/s)+1.
        # But we have specific starts.
        # Let's just loop over the starts for this step.
        for i in range(len(ss)):
            st = ss[i]
            ln = ls[i]
            # sum signs[st : st + ln*s : s]
            total = 0
            for k in range(ln):
                total += signs[st + k * s]
            a = abs(total)
            if a > max_val:
                max_val = a
    return max_val

def _eval_all_fast(signs, starts, steps, lengths):
    """Faster evaluation using numpy slicing where possible."""
    n = len(signs)
    max_val = 0
    unique_steps = np.unique(steps)
    for s in unique_steps:
        mask = steps == s
        ss = starts[mask]
        ls = lengths[mask]
        # For each start, compute sum
        # We can use numpy take with computed indices
        # But creating indices for all APs at once might be memory heavy.
        # Let's try a different approach: for step=1, use cumsum.
        if s == 1:
            cumsum = np.concatenate([[0], np.cumsum(signs)])
            # For each start, length, sum = cumsum[start+length] - cumsum[start]
            ends = ss + ls
            totals = cumsum[ends] - cumsum[ss]
            max_val = max(max_val, int(np.max(np.abs(totals))))
        else:
            # For other steps, we can still vectorize somewhat
            # But the indices are not contiguous.
            # Let's just loop.
            for i in range(len(ss)):
                st = ss[i]
                ln = ls[i]
                total = 0
                for k in range(ln):
                    total += signs[st + k * s]
                a = abs(total)
                if a > max_val:
                    max_val = a
    return max_val

def _eval_all_vectorized(signs, starts, steps, lengths):
    """Fully vectorized evaluation if possible."""
    n = len(signs)
    max_val = 0
    unique_steps = np.unique(steps)
    for s in unique_steps:
        mask = steps == s
        ss = starts[mask]
        ls = lengths[mask]
        if s == 1:
            cumsum = np.concatenate([[0], np.cumsum(signs)])
            ends = ss + ls
            totals = cumsum[ends] - cumsum[ss]
            max_val = max(max_val, int(np.max(np.abs(totals))))
        else:
            # Build index array
            # For each AP i, indices are ss[i] + k*s for k in range(ls[i])
            # This is tricky to vectorize without creating huge arrays.
            # Let's stick to the loop for s > 1.
            for i in range(len(ss)):
                st = ss[i]
                ln = ls[i]
                total = 0
                for k in range(ln):
                    total += signs[st + k * s]
                a = abs(total)
                if a > max_val:
                    max_val = a
    return max_val

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Precompute AP indices
    starts, steps, lengths = _build_ap_indices(n)
    num_aps = len(starts)

    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int8)

    # Compute initial objective
    current_val = _eval_all_vectorized(signs, starts, steps, lengths)
    best_val = current_val
    best_signs = signs.copy()

    # Simulated annealing
    T = 10.0
    cooling = 0.995
    min_T = 0.1

    while time.time() < deadline and T > min_T:
        # Pick a random position to flip
        i = rng.randint(n)
        old_sign = signs[i]
        signs[i] = -old_sign

        # Compute new objective
        new_val = _eval_all_vectorized(signs, starts, steps, lengths)

        # Accept or reject
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = signs.copy()
        else:
            # Reject with probability exp(-(new_val - current_val) / T)
            delta = new_val - current_val
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_val = new_val
                else:
                    signs[i] = old_sign  # Revert
            else:
                current_val = new_val

        T *= cooling

    # Final local search to polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_sign = signs[i]
            signs[i] = -old_sign
            new_val = _eval_all_vectorized(signs, starts, steps, lengths)
            if new_val < current_val:
                current_val = new_val
                improved = True
                if new_val < best_val:
                    best_val = new_val
                    best_signs = signs.copy()
            else:
                signs[i] = old_sign

    # Return best signs as a list of Python integers
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
