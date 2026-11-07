"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 12
    m = 40
    
    # We want to maximize the number of unordered pairs (A, B) such that A is a strict subset of B.
    # This is equivalent to finding a set of 40 subsets of {0..11} that maximizes the number of comparable pairs.
    # The maximum possible number of comparable pairs in a poset of size m is m*(m-1)/2 if it's a chain.
    # Since we can pick any 40 subsets from the 2^12 = 4096 possible subsets, and the power set of a 12-element set
    # contains chains of length 13 (from empty set to full set), we can pick a long chain.
    # A chain of length 13 gives 13*12/2 = 78 comparable pairs.
    # But we need 40 sets. If we pick a chain of length 40, the number of comparable pairs is 40*39/2 = 780.
    # Is it possible to have a chain of length 40 in the power set of a 12-element set?
    # The maximum chain length in the power set of an n-element set is n+1.
    # So for n=12, the maximum chain length is 13.
    # Therefore, we cannot have a chain of length 40.
    
    # We need to find a subset of 40 elements in the poset of subsets of {0..11} that maximizes the number of comparable pairs.
    # This is a known problem. The optimal structure is not a single chain, but a union of chains or a more complex structure.
    # However, a simple greedy approach or a known result might help.
    
    # Let's think about the structure. To maximize comparable pairs, we want the sets to be as "nested" as possible.
    # One effective strategy is to take all subsets of a certain size and some of adjacent sizes, but that doesn't maximize comparability within the same size (no two sets of the same size are comparable unless they are equal).
    # So we should avoid having too many sets of the same size.
    # A good heuristic is to take a long chain and then add sets that are comparable to as many existing sets as possible.
    
    # Let's try a greedy algorithm:
    # 1. Start with an empty list of sets.
    # 2. At each step, add the set S (from the 4096 possible) that maximizes the number of comparable pairs with the already selected sets.
    # 3. Tie-breaking: prefer smaller sets or larger sets? Or just arbitrary.
    
    # However, 4096 is small enough that we can precompute the comparability graph or just use a simple greedy with some heuristics.
    # But 60 seconds is a generous time limit. A greedy algorithm with 40 iterations, each scanning 4096 candidates, is 40 * 4096 = 163,840 operations, which is very fast.
    
    # Let's implement the greedy algorithm.
    
    # Precompute all subsets as bitmasks for efficient comparison.
    # subset A is a strict subset of B if (A & B) == A and A != B.
    
    all_subsets = list(range(1 << n))
    
    selected = []
    # We will maintain a list of selected masks.
    
    # To speed up, we can precompute for each mask, the list of masks that are comparable to it?
    # No, let's just do it directly.
    
    # Greedy step:
    for _ in range(m):
        best_mask = -1
        best_score = -1
        # For each candidate mask not yet selected:
        for mask in all_subsets:
            if mask in selected:
                continue
            # Count how many selected sets are comparable to this mask.
            score = 0
            for sel_mask in selected:
                # Check if mask is a strict subset of sel_mask or vice versa.
                if (mask & sel_mask) == mask and mask != sel_mask:
                    score += 1
                elif (mask & sel_mask) == sel_mask and mask != sel_mask:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
    
    # Convert selected masks back to lists of integers.
    result = []
    for mask in selected:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
