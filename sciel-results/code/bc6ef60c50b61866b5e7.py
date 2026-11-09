"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Vertices 0..11
    n = 12
    m = 40
    
    # We want to maximize the number of comparable pairs (A, B) with A ⊂ B.
    # The total number of pairs is C(40, 2) = 780.
    # To maximize comparable pairs, we should form chains.
    # A chain of length L has L-1 comparable pairs per adjacent pair, but actually
    # in a chain, every pair is comparable. So a chain of size k has C(k,2) comparable pairs.
    # To maximize the sum of C(k_i, 2) subject to sum k_i <= 40 and each chain being a valid chain of subsets.
    
    # The longest possible chain of subsets of a 12-element set has length 13 (sizes 0 to 12).
    # We can have at most 13 sets in a single chain.
    
    # Let's try to use multiple chains.
    # If we use one chain of size 13, we get C(13,2) = 78 pairs. Remaining 27 sets.
    # If we use another chain of size 13, we get 78 pairs. Remaining 14 sets.
    # If we use a third chain of size 13, we get 78 pairs. Remaining 1 sets.
    # Total so far: 78 * 3 = 234. Plus the last set adds 0. Total 234.
    
    # Can we do better?
    # What if we use chains of different sizes?
    # The function C(k,2) is convex. So we want chains to be as large as possible.
    # Max chain size is 13.
    # 40 = 13 + 13 + 13 + 1.
    # Sum = 78 + 78 + 78 + 0 = 234.
    
    # Is it possible to have a chain larger than 13? No, because there are only 13 distinct sizes (0..12).
    # And a chain must have distinct sizes.
    
    # What if we don't just use chains, but also exploit comparabilities between different chains?
    # If we have two chains, some elements from one chain might be subsets of elements from the other.
    # However, mixing chains usually reduces the density of comparabilities within the selected sets
    # because you break the pure chain structure.
    # Let's verify if 234 is the maximum.
    # The maximum number of comparable pairs in a family of m subsets of an n-set is bounded.
    # For m=40, n=12.
    # The "star" or "chain" heuristic suggests 234.
    
    # Let's construct 3 chains of size 13 and 1 extra set.
    # Chain 1: sizes 0..12.
    #   S_0 = {}
    #   S_1 = {0}
    #   S_2 = {0,1}
    #   ...
    #   S_12 = {0,1,...,11}
    # Chain 2: sizes 0..12.
    #   To be distinct from Chain 1, we can use a different ordering or different elements.
    #   But wait, the sets must be distinct.
    #   If Chain 1 uses {0}, {0,1}, ..., Chain 2 cannot use {0} again.
    #   So we need to partition the 4096 subsets into chains? No, we just need 40 distinct subsets.
    #   We can pick 13 subsets that form a chain, another 13 disjoint subsets that form a chain, etc.
    #   Are there 3 disjoint chains of length 13?
    #   Yes, easily.
    #   Chain A: based on adding elements 0,1,2...
    #     A_i = {0, ..., i-1} for i=0..12.
    #   Chain B: based on adding elements 1,2,3... but shifted?
    #     We need distinct sets.
    #     B_0 = {1} ? No, B must be a chain.
    #     Let's use different "directions" or starting points.
    #     Actually, we can just pick any 13 sets that form a chain.
    #     Chain A: {0}, {0,1}, {0,1,2}, ..., {0..11} plus empty set?
    #       A = [{}, {0}, {0,1}, {0,1,2}, ..., {0..10}, {0..11}] -> 13 sets.
    #     Chain B: Must be disjoint from A.
    #       Can we find another chain of 13 sets disjoint from A?
    #       The set {1} is not in A (A has {0}, {0,1}... but not {1} alone).
    #       B = [{}, ...] wait, {} is in A. So B cannot use {}.
    #       So B must start from size 1 or higher?
    #       If B starts from size 1, the max size is 12. Length 12.
    #       So we can't have two chains of length 13 if they must be disjoint AND one uses the empty set.
    #       Only one set is the empty set. So only one chain can include it.
    #       Chain A includes {}. Length 13 (sizes 0..12).
    #       Chain B cannot include {}. So max length is 12 (sizes 1..12).
    #       Chain C cannot include {} or any set in A or B.
    
    # Let's refine the chain packing.
    # Chain 1: S_0, S_1, ..., S_12 where S_i has size i.
    #   S_0 = {}
    #   S_1 = {0}
    #   S_2 = {0,1}
    #   ...
    #   S_12 = {0..11}
    #   This uses 13 sets.
    
    # Chain 2: Must be disjoint from Chain 1.
    #   Cannot use {}.
    #   Cannot use {0}, {0,1}, ..., {0..11}.
    #   Can we form a chain of length 12?
    #   T_1 = {1}
    #   T_2 = {1,2}
    #   T_3 = {1,2,3}
    #   ...
    #   T_12 = {1,2,...,11}
    #   Are these distinct from Chain 1?
    #   Chain 1 sets all contain 0 (except {}).
    #   Chain 2 sets do NOT contain 0.
    #   So they are disjoint.
    #   Length 12.
    #   Total sets so far: 13 + 12 = 25.
    #   Comparable pairs: C(13,2) + C(12,2) = 78 + 66 = 144.
    
    # Chain 3: Disjoint from Chain 1 and 2.
    #   Cannot use sets containing only 0s (Chain 1) or only 1,2.. (Chain 2).
    #   Chain 1: subsets of {0..11} that are prefixes of 0,1,2..
    #   Chain 2: subsets of {1..11} that are prefixes of 1,2,3..
    #   Let's try Chain 3 starting with {2}.
    #   U_1 = {2}
    #   U_2 = {2,3}
    #   ...
    #   U_11 = {2,3,...,11}
    #   Length 11.
    #   Disjoint from Chain 1 (which have 0)? Yes, if we ensure they don't have 0.
    #   Disjoint from Chain 2 (which have 1)? Yes, if we ensure they don't have 1.
    #   So U_i = {2, 3, ..., i+1}.
    #   These sets do not contain 0 or 1.
    #   Chain 1 sets contain 0 (except {}).
    #   Chain 2 sets contain 1 (except if size 0, but Chain 2 starts at size 1).
    #   So Chain 3 is disjoint from Chain 1 and 2.
    #   Total sets: 25 + 11 = 36.
    #   Comparable pairs: 144 + C(11,2) = 144 + 55 = 199.
    
    # Chain 4: Disjoint from 1, 2, 3.
    #   Start with {3}.
    #   V_1 = {3}
    #   ...
    #   V_10 = {3,4,...,11}
    #   Length 10.
    #   Disjoint from previous?
    #   Previous sets have 0 (C1), 1 (C2), 2 (C3).
    #   V sets have 3..11. No 0,1,2.
    #   So disjoint.
    #   Total sets: 36 + 10 = 46. Too many. We only need 40.
    #   We have 36 sets. We need 4 more.
    #   Take 4 sets from Chain 4.
    #   V_1, V_2, V_3, V_4.
    #   These form a chain of length 4.
    #   Comparable pairs: 199 + C(4,2) = 199 + 6 = 205.
    
    # Total comparable pairs: 205.
    # Is this better than the seed?
    # Seed: [[b for b in range(12) if (mask>>b)&1] for mask in range(40)]
    # This generates sets for masks 0..39.
    # Mask 0: []
    # Mask 1: [0]
    # Mask 2: [1]
    # Mask 3: [0,1]
    # ...
    # This is not a chain. It's the first 40 binary numbers.
    # Let's estimate the seed score.
    # Pairs (i,j) with i<j such that S_i subset S_j.
    # This is likely much lower than 205.
    
    # Can we do better than 205?
    # We used chains of lengths 13, 12, 11, 4.
    # Sum of lengths = 40.
    # Sum of C(L,2) = 78 + 66 + 55 + 6 = 205.
    
    # What if we use lengths 13, 13, 13, 1?
    # We established we can't have two chains of length 13 because of the empty set.
    # Only one chain can have the empty set.
    # So max chain lengths: 13, 12, 11, 10, 9, ...
    # We want to maximize sum C(L_i, 2) subject to sum L_i = 40 and L_i <= 13-i (roughly, due to disjointness constraints on 'starting elements').
    # Actually, the constraint is that we need disjoint chains.
    # Chain k starts with element k-1?
    # Chain 1: starts with 0 (and has empty set). Max len 13.
    # Chain 2: starts with 1. Max len 12.
    # Chain 3: starts with 2. Max len 11.
    # Chain 4: starts with 3. Max len 10.
    # Chain 5: starts with 4. Max len 9.
    # ...
    # Chain 12: starts with 11. Max len 1.
    
    # We have 40 slots.
    # Greedy allocation:
    # Take max possible from Chain 1: 13. Rem 27.
    # Take max possible from Chain 2: 12. Rem 15.
    # Take max possible from Chain 3: 11. Rem 4.
    # Take max possible from Chain 4: min(10, 4) = 4. Rem 0.
    # Lengths: 13, 12, 11, 4.
    # Score: 78 + 66 + 55 + 6 = 205.
    
    # Is there a better combination?
    # What if we take less from Chain 3 to allow more from Chain 4?
    # C(11,2)=55. C(10,2)=45.
    # If we reduce Chain 3 by 1 (len 10), we have 5 left.
    # Chain 4 takes 5.
    # Lengths: 13, 12, 10, 5.
    # Score: 78 + 66 + 45 + 10 = 199. Worse.
    
    # What if we reduce Chain 2?
    # 13, 11, 11, 5?
    # Chain 2 max 12. If we take 11, rem 16.
    # Chain 3 max 11. Take 11. Rem 5.
    # Chain 4 take 5.
    # Lengths: 13, 11, 11, 5.
    # Score: 78 + 55 + 55 + 10 = 198. Worse.
    
    # It seems 13, 12, 11, 4 is optimal for this "prefix" chain structure.
    
    # Let's implement this specific construction.
    
    sets = []
    
    # Chain 1: Length 13
    # S_0 = {}
    # S_i = {0, 1, ..., i-1} for i=1..12
    for i in range(13):
        sets.append(list(range(i)))
        
    # Chain 2: Length 12
    # T_i = {1, 2, ..., i} for i=1..12
    # Note: T_1={1}, T_2={1,2}...
    for i in range(1, 13):
        sets.append(list(range(1, i+1)))
        
    # Chain 3: Length 11
    # U_i = {2, 3, ..., i+1} for i=1..11
    for i in range(1, 12):
        sets.append(list(range(2, i+2)))
        
    # Chain 4: Length 4
    # V_i = {3, 4, ..., i+2} for i=1..4
    for i in range(1, 5):
        sets.append(list(range(3, i+3)))
        
    # Check distinctness
    # Chain 1: sets containing 0 (except first). Sizes 0..12.
    # Chain 2: sets containing 1, not 0. Sizes 1..12.
    # Chain 3: sets containing 2, not 0,1. Sizes 1..11.
    # Chain 4: sets containing 3, not 0,1,2. Sizes 1..4.
    
    # Are any sets duplicated?
    # C1 sets: {0..i-1}. All contain 0 (except empty).
    # C2 sets: {1..i}. All contain 1, no 0.
    # C3 sets: {2..i+1}. All contain 2, no 0,1.
    # C4 sets: {3..i+2}. All contain 3, no 0,1,2.
    
    # Since the "first element" (min element) is different for each chain (0 for C1 non-empty, 1 for C2, 2 for C3, 3 for C4),
    # and C1 has the empty set (min undefined/None), all sets are distinct.
    
    # Verify count
    assert len(sets) == 40, f"Got {len(sets)} sets"
    
    # Verify types
    for s in sets:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
    
    return sets

# EVOLVE-BLOCK-END
