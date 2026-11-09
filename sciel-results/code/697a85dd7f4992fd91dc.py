"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute inverses for denominators 1..10 modulo 101
    P = 101
    max_denom = 10
    inverses = [0] * (max_denom + 1)
    for i in range(1, max_denom + 1):
        inverses[i] = pow(i, -1, P)
    
    # Precompute sums for all non-empty subsets of {1, ..., 10}
    # Map residue -> list of (length, subset_list)
    residue_map = {r: [] for r in range(P)}
    
    # There are 2^10 - 1 = 1023 non-empty subsets
    for mask in range(1, 1 << max_denom):
        subset = []
        s = 0
        for i in range(1, max_denom + 1):
            if mask & (1 << (i - 1)):
                subset.append(i)
                s = (s + inverses[i]) % P
        residue_map[s].append((len(subset), subset))
        
    # Find the minimal k such that all residues have a representation of size <= k
    min_k = max_denom
    for k in range(1, max_denom + 1):
        all_covered = True
        for r in range(P):
            has_small_rep = False
            for length, subset in residue_map[r]:
                if length <= k:
                    has_small_rep = True
                    break
            if not has_small_rep:
                all_covered = False
                break
        if all_covered:
            min_k = k
            break
            
    # Construct the final result using subsets of size <= min_k
    # To be deterministic and minimal, pick the shortest subset for each residue
    result = []
    for r in range(P):
        # Find the shortest subset for this residue
        best_subset = None
        best_len = max_denom + 1
        for length, subset in residue_map[r]:
            if length < best_len:
                best_len = length
                best_subset = subset
        # Since min_k was found, best_subset must exist and best_len <= min_k
        result.append(best_subset)
        
    return result
# EVOLVE-BLOCK-END
