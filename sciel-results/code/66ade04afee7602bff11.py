import time
import itertools

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    start_time = time.time()
    DEADLINE = 55  # seconds, safety margin before 60s kill
    
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # Precompute all subsets grouped by length
    # subsets_by_len[k] = list of (tuple_of_denoms, sum_of_inverses)
    subsets_by_len = [[] for _ in range(MAX_DENOM + 1)]
    
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            subsets_by_len[length].append((combo, s))
    
    # Since denominators may recur across different rows but never within one row,
    # and subsets inherently have distinct elements, the problem decomposes per-residue.
    # For each residue r, find the shortest subset whose inverse-sum equals r mod 101.
    # The optimal max row length is the maximum over all residues of their min subset length.
    
    # For each residue, find the minimum length and a corresponding subset.
    best_combo = [None] * MOD
    min_len_for_residue = [None] * MOD
    
    for length in range(1, MAX_DENOM + 1):
        for combo, s in subsets_by_len[length]:
            if min_len_for_residue[s] is None:
                min_len_for_residue[s] = length
                best_combo[s] = list(combo)
    
    # Fallback: if any residue is uncovered (shouldn't happen), assign any valid subset
    for r in range(MOD):
        if best_combo[r] is None:
            for length in range(1, MAX_DENOM + 1):
                for combo, s in subsets_by_len[length]:
                    if s == r:
                        best_combo[r] = list(combo)
                        min_len_for_residue[r] = length
                        break
                if best_combo[r] is not None:
                    break
            if best_combo[r] is None:
                best_combo[r] = [1]
    
    # Verify correctness
    for r in range(MOD):
        total = sum(inverses[d] for d in best_combo[r]) % MOD
        if total != r:
            raise ValueError(f"Residue {r}: expected {r}, got {total} from {best_combo[r]}")
        if len(set(best_combo[r])) != len(best_combo[r]):
            raise ValueError(f"Residue {r}: non-distinct denominators in {best_combo[r]}")
    
    return best_combo
