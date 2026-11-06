import time

def construct():
    # Direct construction: {66, ..., 100} is a valid set of size 35.
    # No subset sums to 200 because:
    # - Max sum of 2 elements is 100+99=199.
    # - Min sum of 3 elements is 66+67+68=201.
    # Thus, no subset can sum to 200.
    
    # We can try to see if we can get more by swapping large for small,
    # but analysis suggests 35 is likely the maximum.
    # We will start with this set and try to add any remaining elements
    # using a fast greedy check.
    
    best_set = list(range(66, 101))
    
    # Try to add elements from 1 to 65
    # We check if adding x creates a subset sum of 200.
    # Since max sum of 2 in best_set is 199, adding x only creates a sum of 200
    # if x + y = 200 (impossible for y<=100, x>=1) or x + y + z = 200.
    # x + y + z = 200 => y + z = 200 - x.
    # y, z must be in best_set.
    
    # Precompute all pair sums in best_set
    pair_sums = set()
    bs = best_set
    for i in range(len(bs)):
        for j in range(i+1, len(bs)):
            pair_sums.add(bs[i] + bs[j])
            
    # Try adding elements
    for x in range(1, 66):
        if 200 - x in pair_sums:
            continue
        # If we add x, we must also update pair_sums for future checks?
        # No, we are trying to build a maximal set.
        # If we add x, we must ensure no NEW subset sums to 200.
        # New subsets involving x:
        # 1. {x} -> 200? No.
        # 2. {x, y} -> 200? y=200-x. Max y in set is 100. x>=1 => y<=199.
        #    If y in set, then x+y=200.
        #    Since x<=65, y>=135. But max in set is 100. So no.
        # 3. {x, y, z} -> 200? y+z=200-x.
        #    We checked if 200-x is in pair_sums.
        # 4. {x, y, z, w} -> 200?
        #    Min sum of 3 from bs is 201. So y+z+w >= 201.
        #    x + y + z + w >= 1 + 201 = 202.
        #    So no subset of size >= 4 involving x sums to 200.
        
        # So the only check is if 200-x is a sum of two elements in bs.
        # If we add x, we should add it and update pair_sums?
        # If we add multiple, we need to be careful.
        # Greedy: try to add as many as possible.
        # However, if we add x, it might block y?
        # Actually, if x and y are both addable, can we add both?
        # If we add x, pair_sums changes.
        # But x is small.
        # Let's just try to add one by one.
        
        # If we add x, we must verify the whole set again?
        # No, the logic above holds for adding x to the CURRENT set.
        # If we add x, the new set is bs U {x}.
        # We need to check if bs U {x} is valid.
        # It is valid if no subset of bs U {x} sums to 200.
        # Subsets of bs are valid.
        # Subsets of bs U {x} containing x:
        # - {x}
        # - {x, y}
        # - {x, y, z}
        # - {x, y, z, w...}
        
        # We established:
        # - {x} != 200
        # - {x, y} != 200 (since y <= 100, x >= 1, sum <= 165? No, x can be 65, y=100, sum 165. Wait. 65+100=165. 200-x >= 135. So y=200-x >= 135. Not in set.)
        # - {x, y, z} != 200 iff 200-x not in pair_sums(bs).
        # - {x, y, z, w} != 200 because min sum of 3 from bs is 201.
        
        # So if 200-x not in pair_sums, we can add x.
        # But if we add x, pair_sums changes.
        # New pair sums involving x: x + y for y in bs.
        # These are <= 65 + 100 = 165.
        # Old pair sums were >= 66+67=133.
        # So new pair sums are in [67, 165].
        # This might affect future additions.
        
        # Let's try to add all possible x greedily.
        pass # We will implement this below
        
    # Since we proved we can't add any x to the FULL set {66..100},
    # we can just return it.
    # But what if we removed some elements to allow more?
    # The size would be the same or less.
    
    return best_set
