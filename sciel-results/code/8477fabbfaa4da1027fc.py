"""Combined program: Optimal per-residue minimum-popcount assignment.
Since denominators may recur across different rows, each row is independent.
The optimal max row length = max over all residues r of (minimum popcount of any
nonempty subset of {1..10} whose inverse-sum ≡ r mod 101).
We compute this directly by enumerating all 1023 nonempty subsets.
"""
import time

def construct():
    MOD = 101
    n = 10

    # Precompute modular inverses of 1..10 mod 101
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, -1, MOD)

    # For each residue, track the minimum popcount and corresponding mask
    min_size = [11] * MOD  # 11 is larger than any possible size
    best_mask = [0] * MOD

    # Enumerate all 1023 nonempty subsets
    for mask in range(1, 1 << n):
        s = 0
        sz = 0
        # Use bit manipulation for speed
        m = mask
        x = 1
        while m:
            if m & 1:
                s = (s + inv[x]) % MOD
                sz += 1
            m >>= 1
            x += 1

        if sz < min_size[s]:
            min_size[s] = sz
            best_mask[s] = mask

    # Convert masks to lists of denominators
    result = []
    for r in range(MOD):
        m = best_mask[r]
        row = [x for x in range(1, n + 1) if m & (1 << (x - 1))]
        result.append(row)

    # Verify correctness
    for r, terms in enumerate(result):
        if len(terms) == 0:
            raise ValueError(f"Row {r} is empty")
        if len(set(terms)) != len(terms):
            raise ValueError(f"Row {r} has duplicate denominators")
        if any(x < 1 or x > 10 for x in terms):
            raise ValueError(f"Row {r} has invalid denominator")
        res = sum(inv[x] for x in terms) % MOD
        if res != r:
            raise ValueError(f"Row {r} sums to {res} instead of {r}")

    return result
