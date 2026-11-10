import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    N = 1000
    MAXV = 2000  # max possible sum (1000+1000)

    def collision_vector(S):
        """Return an array c where c[s] = number of pairs (a,b) in S, a<=b, with a+b=s."""
        ind = np.zeros(MAXV + 1, dtype=np.int32)
        ind[S] = 1
        # autocorrelation via FFT: (ind * ind_rev) gives counts of ordered pairs
        # For a<=b we need: count of pairs (a,b) with a<=b and a+b=s.
        # Total ordered pairs with sum s = (ind * ind_rev)[s] where rev is reversed.
        # Let's compute: for each s, number of (a,b) in S x S with a+b=s is the
        # coefficient of the convolution ind * ind.
        # Using FFT:
        n = MAXV + 1
        size = 1
        while size < 2 * n - 1:
            size *= 2
        F = np.fft.rfft(ind, size)
        G = np.fft.irfft(F * F, size)[:n]
        # G[s] = number of ordered pairs (a,b) in S x S with a+b = s
        # For a<=b: if a<b, both (a,b) and (b,a) counted, so pairs with a<b = (G[s] - diag)/2
        # where diag = 1 if s is even and s/2 in S else 0 (the a=b case)
        # total a<=b reps = (G[s] - diag_s) // 2 + diag_s = (G[s] + diag_s) // 2
        diag = np.zeros(n, dtype=np.int64)
        even_s = np.arange(0, n, 2)
        half = even_s // 2
        diag[even_s] = ind[half]
        reps = (G.astype(np.int64) + diag) // 2
        return reps

    def total_violation(reps):
        """Sum of max(0, reps[s]-2) over all s, i.e., how much we exceed the limit."""
        return int(np.sum(np.maximum(0, reps - 2)))

    def is_valid(reps):
        return bool(np.all(reps <= 2))

    rng = np.random.default_rng(42)

    best_set = None
    best_size = 0

    # Strategy: try multiple random initializations with different densities,
    # then do local search to fix violations and try to add more elements.

    for trial in range(60):
        if time.time() > deadline:
            break

        # Random initialization: pick ~700 random distinct integers
        target_size = 700
        S = set(rng.choice(np.arange(1, N + 1), size=target_size, replace=False).tolist())
        S_arr = np.array(sorted(S), dtype=np.int32)

        reps = collision_vector(S_arr)
        viol = total_violation(reps)

        # Local search phase 1: remove elements to fix violations
        # Phase 2: try to add elements
        # Phase 3: try swaps (remove one, add one) to increase size while staying valid

        # We'll do iterative improvement
        for iteration in range(2000):
            if time.time() > deadline:
                break

            reps = collision_vector(S_arr)
            viol = total_violation(reps)

            if viol == 0:
                # Valid! Try to add more elements
                # Find candidates that can be added without causing violation
                in_set = np.zeros(N + 1, dtype=bool)
                in_set[S_arr] = True
                candidates = np.where(~in_set)[0]
                candidates = candidates[candidates >= 1]

                added = False
                # Shuffle candidates for randomness
                rng.shuffle(candidates)
                for c in candidates:
                    # Check if adding c keeps validity
                    new_S = np.concatenate([S_arr, [c]])
                    new_reps = collision_vector(new_S)
                    if is_valid(new_reps):
                        S_arr = new_S
                        S.add(c)
                        added = True
                        break
                if not added:
                    # Try swaps: remove a low-value element, add a high-value one
                    # Find elements whose removal helps most
                    # For each element e in S, compute how much violation decreases if removed
                    # This is expensive; instead try random swaps
                    swapped = False
                    for _ in range(20):
                        # Pick a random element to remove
                        idx = rng.integers(0, len(S_arr))
                        e = S_arr[idx]
                        # Pick a random candidate to add
                        if len(candidates) == 0:
                            break
                        c = candidates[rng.integers(0, len(candidates))]
                        new_S = np.delete(S_arr, idx)
                        new_S = np.append(new_S, c)
                        new_S = np.sort(new_S)
                        new_reps = collision_vector(new_S)
                        if is_valid(new_reps):
                            S_arr = new_S
                            S.discard(e)
                            S.add(c)
                            # Remove c from candidates, add e
                            in_set[c] = True
                            in_set[e] = False
                            candidates = np.where(~in_set)[0]
                            candidates = candidates[candidates >= 1]
                            swapped = True
                            break
                    if not swapped:
                        break
            else:
                # Has violations: remove the element that causes the most excess
                # For each element e, compute its "blame": sum of (reps[s]-2) for s where e participates
                # Approximate: remove the element that appears in the most over-represented sums
                # Compute for each element e, the number of sums s with reps[s]>2 where e is in some pair
                # This is complex; use heuristic: remove element with highest "load"
                # Load of e = number of other elements f such that e+f has reps>2
                # Simpler: just remove a random element that's in a violating sum

                # Find violating sums
                viol_sums = np.where(reps > 2)[0]
                if len(viol_sums) == 0:
                    break

                # For each element in S, count how many violating sums it participates in
                # An element e participates in sum s if there exists f in S with e+f=s and f>=e
                # Let's compute: for each e, the set of sums it's in is {e+f : f in S, f>=e}
                # We want to find e that maximizes overlap with viol_sums
                # This is O(|S|^2) per iteration which is too slow for |S|~700
                # Instead: pick a random violating sum, pick a random pair in it, remove one

                s = viol_sums[rng.integers(0, len(viol_sums))]
                # Find all pairs (a,b) in S with a<=b and a+b=s
                # a ranges from max(1, s-1000) to min(1000, s//2)
                a_min = max(1, s - N)
                a_max = min(N, s // 2)
                pairs = []
                for a in range(a_min, a_max + 1):
                    b = s - a
                    if a <= b and a in S and b in S:
                        pairs.append((a, b))
                if pairs:
                    a, b = pairs[rng.integers(0, len(pairs))]
                    # Remove one of them (randomly, or the one with higher "load")
                    remove = a if rng.random() < 0.5 else b
                    S.discard(remove)
                    S_arr = np.array(sorted(S), dtype=np.int32)

        # After local search, check final validity
        reps = collision_vector(S_arr)
        if is_valid(reps):
            if len(S_arr) > best_size:
                best_size = len(S_arr)
                best_set = S_arr.copy()

    # Final greedy addition pass on best_set
    if best_set is not None:
        reps = collision_vector(best_set)
        in_set = np.zeros(N + 1, dtype=bool)
        in_set[best_set] = True
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if in_set[x]:
                continue
            new_S = np.append(best_set, x)
            new_reps = collision_vector(new_S)
            if is_valid(new_reps):
                best_set = new_S
                in_set[x] = True

    if best_set is None:
        # Fallback: return the seed greedy result
        chosen, counts = [], {}
        for x in range(1, 1001):
            new = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            for s in new:
                trial[s] = trial.get(s, 0) + 1
            if max(trial[s] for s in new) <= 2:
                chosen.append(x)
                counts = trial
        return chosen

    return list(best_set)
# EVOLVE-BLOCK-END
