import itertools
import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))

    # Precompute modular inverses
    inverses = [0] * (MAX_DENOM + 1)
    for d in denominators:
        inverses[d] = pow(d, -1, MOD)

    # best_combo[r] will store the list of denominators for residue r
    # min_len[r] will store the length of the best combo found so far
    best_combo = [None] * MOD
    min_len = [float('inf')] * MOD

    # We iterate by subset length to ensure we find the minimum length for each residue
    # as soon as possible.
    # Since we want to minimize the MAX row length, finding the min for each row independently
    # is the correct strategy because rows are independent (denominators can be reused).
    
    # Optimization: Precompute sums for all subsets? 
    # There are 2^10 = 1024 subsets. This is very small.
    # We can just iterate through all subsets once, group by sum, and keep the min length.
    # This is faster than iterating by length and checking all residues each time.
    
    # subset_sums[s] = min_length_subset_for_sum_s
    # Actually, we just need the min length for each residue.
    
    min_len_for_residue = [float('inf')] * MOD
    best_subset_for_residue = [None] * MOD

    # Iterate over all non-empty subsets
    # There are 1023 subsets.
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            if length < min_len_for_residue[s]:
                min_len_for_residue[s] = length
                best_subset_for_residue[s] = list(combo)

    # Construct the final result
    result = []
    max_len_found = 0
    for r in range(MOD):
        if best_subset_for_residue[r] is not None:
            result.append(best_subset_for_residue[r])
            if len(best_subset_for_residue[r]) > max_len_found:
                max_len_found = len(best_subset_for_residue[r])
        else:
            # Fallback: should not happen for mod 101 with 10 denominators
            # But if it does, we need a valid subset.
            # This case is theoretically impossible given 1023 subsets and 101 residues,
            # but we handle it gracefully.
            result.append([1]) 
            if 1 > max_len_found:
                max_len_found = 1

    # Verification
    for r in range(MOD):
        total = sum(inverses[d] for d in result[r]) % MOD
        if total != r:
            raise ValueError(f"Verification failed for residue {r}")
        if len(set(result[r])) != len(result[r]):
            raise ValueError(f"Non-distinct denominators for residue {r}")

    return result
