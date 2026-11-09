"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import itertools
import time

def construct():
    n = 12
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for m in range(1, total_masks):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute subset/superset relations using bit tricks
    # For each mask, store list of masks that are subsets of it
    # and masks that are supersets of it
    # Actually, let's think about this differently.
    
    # We want to maximize comparable pairs.
    # Key insight: if we pick all subsets of a fixed set of size k, 
    # we get 2^k - 1 sets, all pairwise comparable.
    # With k=6, we get 63 sets, but we only need 40.
    # So pick a 6-element base set B, and take 40 subsets of B.
    # All 40 are pairwise comparable → C(40,2) = 780 comparable pairs.
    
    # But wait - can we do better? 
    # If we use two disjoint bases, we lose comparability between them.
    # So one chain is best.
    
    # Actually, let's think: a chain of length L has C(L,2) comparable pairs.
    # Max chain in 2^12 is 13 (sizes 0 through 12).
    # So max chain length is 13, giving C(13,2)=78.
    
    # But we don't need a chain! We need comparable pairs, not a total order.
    # The best structure is a union of chains or a "broom" shape.
    
    # Actually, the optimal is to take all subsets of a fixed k-set.
    # For k=6: 2^6=64 subsets, pick 40, all comparable → 780.
    # For k=5: 2^5=32 subsets, pick 40? Can't, only 32.
    # So k=6 with 40 subsets of a 6-element set gives 780.
    
    # Can we do better by mixing?
    # Suppose we take 40 subsets of a 6-set. That's 780.
    # What if we take 39 subsets of one 6-set and 1 subset of another 6-set 
    # that is comparable with many of the first?
    # The extra set would be comparable with at most 39, adding 39 to 780*... 
    # No wait, the 39 subsets among themselves give C(39,2)=741.
    # Plus 39 from the new set = 780. Same.
    
    # So 780 seems to be the max if we use a single base set.
    
    # But what about using a larger base? 
    # If we take subsets of a 7-set, we have 128 subsets.
    # Pick 40 of them. If they're all comparable, that's a chain of 40.
    # But max chain in 2^7 is 8 (sizes 0-7). So we can't have 40 all comparable
    # within a 7-set unless they form a chain, which is impossible for 40.
    
    # Wait, I was wrong. Subsets of a fixed set are NOT all pairwise comparable.
    # {1,2} and {2,3} are both subsets of {1,2,3} but neither is a subset of the other.
    
    # So my earlier claim was wrong. Let me reconsider.
    
    # Two sets A and B are comparable if A⊆B or B⊆A.
    # The number of comparable pairs in a family F is what we want to maximize.
    
    # Known result: The maximum number of comparable pairs in a family of 
    # k subsets of [n] is achieved by taking a chain plus... 
    # Actually, by the result of Kleitman, the maximum is achieved by 
    # taking all subsets of size ≤ some threshold, or something like that.
    
    # Let me think small. For n=12, k=40.
    # A chain of length 13 (all sizes 0-12) gives C(13,2)=78.
    # If I take all subsets of size 0,1,2,...,m for some m, 
    # the number of comparable pairs is sum over pairs where one is subset of other.
    
    # Actually, let's just do a smart local search with good initialization.
    
    # Better initialization: take a chain of length 13, then add sets that 
    # are comparable with many existing sets.
    
    # Even better: use the fact that if we take all subsets of a fixed set S
    # of size s, the number of comparable pairs depends on the structure.
    
    # Let me just do a robust local search with multiple restarts.
    
    def score_family(family):
        count = 0
        m = len(family)
        for i in range(m):
            for j in range(i + 1, m):
                a = family[i]
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
    
    # Precompute comparability matrix for speed
    # comp[a][b] = 1 if a and b are comparable, 0 otherwise
    # But that's 4096*4096 = 16M entries, too much.
    # Instead, compute on the fly but efficiently.
    
    # Let's precompute for each mask, the list of masks it's comparable with
    # Actually, let's just compute the score incrementally.
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Initialize with a good structure:
    # Take all subsets of {0,1,2,3,4,5} that have size 0-3 (to get a nice structure)
    # Actually, let's take a "broom": a long chain plus sets hanging off it.
    
    # Chain: ∅, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11} → 13 sets
    # Then add sets like {0,1,...,i, j} for various j not in {0..i}
    
    # Let's try: start with all subsets of {0,1,2,3,4} (32 sets), 
    # then add 8 more that are comparable with many of these.
    
    # Subsets of {0,1,2,3,4}: 32 sets.
    # Comparable pairs among them: for each pair, comparable if one is subset of other.
    # This is the number of comparable pairs in the boolean lattice of dimension 5.
    # That's sum_{S⊆[5]} (2^{popcount(S)} - 1) / ... 
    # Actually, it's (3^5 - 5*2^5 + 4)/2 = (243-160+4)/2 = 87/2... no.
    # Let me just compute it.
    
    base5 = list(range(32))  # all subsets of {0,1,2,3,4}
    base5_score = score_family(base5)
    
    # Now add 8 more sets. Each new set should be comparable with as many 
    # of the existing 32 as possible.
    # A set T is comparable with all subsets of {0,1,2,3,4} iff 
    # T is a subset of {0,1,2,3,4} or {0,1,2,3,4} is a subset of T.
    # The first case: T is already in our family.
    # The second case: {0,1,2,3,4} ⊆ T. So T contains all of 0-4.
    # T could be {0,1,2,3,4,5}, {0,1,2,3,4,6}, etc.
    # Each such T is comparable with all 32 subsets of {0,1,2,3,4}.
    # So we can add up to 7 more such sets (to get to 40 total).
    
    # Add {0,1,2,3,4,5}, {0,1,2,3,4,6}, {0,1,2,3,4,7}, 
    # {0,1,2,3,4,8}, {0,1,2,3,4,9}, {0,1,2,3,4,10}, {0,1,2,3,4,11}
    # That's 7 more, total 39. Need one more.
    
    # The 40th set: it should be comparable with many of the 39.
    # Option: {0,1,2,3,4,5,6} - comparable with all 32 subsets of {0-4} 
    # and with {0,1,2,3,4,5} and {0,1,2,3,4,6}. 
    # Is it comparable with {0,1,2,3,4,7}? {0,1,2,3,4,5,6} ∩ {0,1,2,3,4,7} 
    # = {0,1,2,3,4}, which is not equal to either. So not comparable.
    # So it's comparable with 32 + 2 = 34 of the 39.
    
    # Alternative: just do local search from this initialization.
    
    random.seed(123)
    
    # Build initial family
    initial = list(base5)
    for extra in range(5, 12):
        initial.append((1 << 5) - 1 | (1 << extra))  # {0,1,2,3,4} ∪ {extra}
    # That's 32 + 7 = 39 sets. Add one more.
    initial.append((1 << 6) - 1)  # {0,1,2,3,4,5}
    
    # Wait, {0,1,2,3,4,5} is already in initial (when extra=5).
    # Let me redo.
    
    initial = list(range(32))  # 32 subsets of {0,1,2,3,4}
    base_mask = (1 << 5) - 1  # 0b0011111 = {0,1,2,3,4}
    for extra in range(5, 11):  # 5,6,7,8,9,10 → 6 sets
        initial.append(base_mask | (1 << extra))
    # 32 + 6 = 38. Need 2 more.
    initial.append(base_mask | (1 << 11))  # {0,1,2,3,4,11}
    initial.append(base_mask | (1 << 5) | (1 << 6))  # {0,1,2,3,4,5,6}
    
    # That's 40 sets.
    
    current_family = initial[:]
    current_score = score_family(current_family)
    
    # Local search: try replacing each element
    improved = True
    max_iterations = 500
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(current_family)):
            old_set = current_family[i]
            
            # Compute current contribution of old_set
            C_old = 0
            for j in range(len(current_family)):
                if i == j:
                    continue
                if is_comparable(old_set, current_family[j]):
                    C_old += 1
            
            best_gain = 0
            best_new_set = -1
            
            for new_set in all_masks:
                if new_set == old_set:
                    continue
                if new_set in current_family:
                    continue
                
                C_new = 0
                for j in range(len(current_family)):
                    if i == j:
                        continue
                    if is_comparable(new_set, current_family[j]):
                        C_new += 1
                
                gain = C_new - C_old
                if gain > best_gain:
                    best_gain = gain
                    best_new_set = new_set
            
            if best_new_set != -1:
                current_family[i] = best_new_set
                current_score += best_gain
                improved = True
    
    # Convert masks to lists
    result = []
    for m in current_family:
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result

# EVOLVE-BLOCK-END
