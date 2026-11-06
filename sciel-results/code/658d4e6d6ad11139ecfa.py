# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    target_size = 80
    
    # Strategy: Use a randomized constructive heuristic.
    # 1. Start with a sparse skeleton of large numbers (e.g., powers of 4 or similar geometric progression)
    #    to ensure unique triple sums easily due to digit separation in base 4.
    # 2. Then greedily insert smaller numbers into the gaps, checking for conflicts.
    # 3. Since the greedy approach in the seed only got 512 (which is actually > 80, wait, the seed returns up to 80? 
    #    No, the seed code loops range(1, 513). If it finds valid ones, it adds them. 
    #    The task says "Maximize cardinality" but "At most 80". 
    #    Wait, the seed code:
    #    `for x in range(1, 513):`
    #    It checks if adding x keeps all triple sums unique.
    #    If the seed score is 1.0, and the task says "search score is value/seed", and seed returns some list.
    #    The evaluator measures "raw value". The task says "Maximize cardinality".
    #    So the value is likely `len(selected)`.
    #    The seed program returns a list. Let's assume the value is the length of the list.
    #    The seed loop goes up to 512. If it finds 80 valid numbers, it stops? No, it doesn't stop.
    #    But the task says "At most 80 positive integers". 
    #    Does the evaluator cap it? Or does the seed only find 80?
    #    Actually, B_3[1] sets are very sparse. 
    #    For n=4096, the maximum size of a B_3[1] set is known to be small.
    #    Let's check the density. The number of triples is k(k+1)(k+2)/6.
    #    The sums range from 3 to 3*4096 = 12288.
    #    So we need k(k+1)(k+2)/6 <= 12288 - 3 + 1 = 12286.
    #    k^3/6 <= 12286 => k^3 <= 73716 => k <= 41.
    #    So the maximum possible cardinality is around 41.
    #    The seed program goes up to 512. It will likely stop finding valid candidates much earlier.
    #    Let's re-read the seed code. It adds x if valid. It doesn't stop at 80.
    #    But the task says "construct() -> 1..80 distinct positive integers".
    #    Maybe the seed returns a list of size ~40?
    #    If the seed score is 1.0, and the value is the size, then the seed achieves the max possible?
    #    Or maybe the seed achieves a specific number, say 35, and the score is 1.0.
    #    If I can find a set of size 36, my score would be 36/35 > 1.0.
    #    
    #    Let's verify the upper bound.
    #    Sums of triples from {a1, ..., ak} with 1 <= a1 < ... < ak <= N.
    #    The sums are distinct.
    #    Min sum = 3*a1 >= 3. Max sum = 3*ak <= 3N.
    #    Number of distinct sums <= 3N - 3 + 1 = 3N - 2.
    #    Number of unordered triples with replacement is C(k+2, 3) = (k+2)(k+1)k/6.
    #    So (k+2)(k+1)k/6 <= 3N - 2.
    #    For N=4096, 3*4096 - 2 = 12286.
    #    k=40: 42*41*40/6 = 11480.
    #    k=41: 43*42*41/6 = 12341 > 12286.
    #    So k cannot be 41. Max k is 40.
    #    
    #    So the theoretical maximum is 40.
    #    The seed program likely finds a set close to this maximum, or maybe less due to greediness.
    #    If the seed finds 35, and I find 40, my score is 40/35 = 1.14.
    #    If the seed finds 40, my score is 1.0.
    #    
    #    The directive says: "Use a randomized constructive heuristic that selects elements from a large geometric progression... to guarantee unique triple sums... then attempts to insert smaller integers into the gaps".
    #    This suggests the seed might not be optimal.
    #    
    #    Let's try to construct a set of size 40.
    #    A known construction for B_h sets uses powers of (2h+1). For h=3, base 7.
    #    But we want to maximize density.
    #    
    #    Alternative approach:
    #    1. Generate a random set of size K (e.g., 40) from [1, 4096].
    #    2. Check if it's a valid B_3[1] set.
    #    3. If not, try to repair or regenerate.
    #    
    #    However, random sets are very likely to have collisions.
    #    
    #    Better approach:
    #    1. Start with a known valid small set (e.g., {1, 2, 4, 8, ...} powers of 2? No, powers of 4 is safer for base 4 separation).
    #    Powers of 4: 1, 4, 16, 64, 256, 1024, 4096. Size 7.
    #    This is very sparse.
    #    
    #    Let's look at the "gaps" idea.
    #    If we have a set S, and we want to add x, we need to ensure that no triple sum involving x collides with any existing triple sum.
    #    Existing triple sums: T.
    #    New triple sums involving x:
    #    - x+x+x = 3x
    #    - x+x+y = 2x+y for y in S
    #    - x+y+z = x+y+z for y,z in S
    #    
    #    We need 3x, 2x+y, x+y+z to not be in T, and also not collide with each other.
    #    
    #    The seed program does exactly this: it iterates x from 1 to 512, and checks if adding x is valid.
    #    Why does it stop? It doesn't stop, but it might not find valid x's after a certain point because the constraints become tight.
    #    Since the max is 40, the seed might find 40 if it's lucky or if the greedy order works well.
    #    However, greedy from 1 upwards might fill up the small numbers, which are more prone to collisions?
    #    Or maybe it fills up the space too densely?
    #    
    #    Let's try a different greedy order:
    #    1. Start with an empty set.
    #    2. Try to add numbers from a large range, but prioritize larger numbers?
    #    Or randomize the order?
    #    
    #    The directive specifically suggests:
    #    "selects elements from a large geometric progression... to guarantee unique triple sums... then attempts to insert smaller integers into the gaps"
    #    
    #    So:
    #    1. Create a skeleton S0 using a geometric progression (e.g., powers of 4, or 4^i * c).
    #       Ensure S0 is a valid B_3[1] set.
    #    2. Try to insert other numbers (smaller ones, or ones in the gaps) into S0 to form S.
    #       Check validity incrementally.
    #    
    #    Let's implement this.
    #    
    #    Step 1: Skeleton
    #    Let base = 4.
    #    S0 = [1, 4, 16, 64, 256, 1024, 4096]
    #    Check if valid. Yes, because in base 4, each element has a single non-zero digit at a different position.
    #    Sum of three such numbers will have digits that are sums of 0s and 1s (or 2s, 3s if same position, but positions are distinct so no carry? 
    #    Wait. If we have 1 (1000000_4), 4 (100000_4), etc.
    #    Triple (1,1,1) -> 3.
    #    Triple (1,1,4) -> 1+1+4 = 6 = 12_4.
    #    Triple (1,4,4) -> 1+4+4 = 9 = 21_4.
    #    Triple (4,4,4) -> 12 = 30_4.
    #    Since each power of 4 is in a distinct "digit" position, and the max sum of three is 3, which is less than 4, there are no carries between digits.
    #    Thus, the base-4 representation of the sum uniquely identifies the multiset of three elements.
    #    So S0 is a valid B_3[1] set.
    #    
    #    Step 2: Insertion
    #    We have S0. We want to add more elements.
    #    We can try to add elements from [1, 4096] \ S0.
    #    We should prioritize elements that are "safe".
    #    The directive says "insert smaller integers into the gaps".
    #    The gaps are between the powers of 4.
    #    E.g., between 1 and 4: 2, 3.
    #    Between 4 and 16: 5..15.
    #    Between 16 and 64: 17..63.
    #    etc.
    #    
    #    Let's try to insert numbers in increasing order of "gap index" or just iterate through all candidates and check.
    #    To maximize the number of inserted elements, we should try to add as many as possible.
    #    Since the check is O(k^2) or O(k^3) depending on implementation, and k is small (~40), we can afford to check many candidates.
    #    
    #    Algorithm:
    #    1. Initialize S with the skeleton S0.
    #    2. Compute the set of all triple sums T for S.
    #    3. Create a list of candidate numbers C = [x for x in range(1, 4097) if x not in S].
    #    4. Shuffle C? Or sort by some heuristic?
    #       The directive says "insert smaller integers into the gaps".
    #       So maybe sort C by value?
    #       Or sort by "distance to nearest skeleton element"?
    #       Let's just try sorting by value (ascending) first, as in the seed, but starting with a valid skeleton.
    #       Actually, the seed starts with empty and goes 1, 2, 3...
    #       Here we start with S0 and add others.
    #       
    #    5. For each candidate x in C:
    #       a. Check if adding x to S keeps all triple sums unique.
    #       b. If yes, add x to S, update T.
    #       c. If no, skip x.
    #    6. Return S.
    #    
    #    Why would this be better than the seed?
    #    The seed starts with 1, 2, 3... and might get stuck in a local minimum where it can't add larger numbers because of conflicts with small ones.
    #    By starting with a sparse, well-separated skeleton, we have more "room" for other numbers.
    #    The skeleton elements are far apart, so their triple sums are spread out.
    #    This might allow us to pack more numbers in the gaps.
    #    
    #    Let's refine the conflict check.
    #    To check if x can be added to S:
    #    - New sums involving x:
    #      - 3x
    #      - 2x + s for s in S
    #      - x + s1 + s2 for s1, s2 in S (unordered, with replacement)
    #    - We need to ensure:
    #      1. None of these new sums are in T (existing sums).
    #      2. None of these new sums collide with each other.
    #    
    #    Implementation:
    #    - Create a set `new_sums`.
    #    - Add 3x.
    #    - For each s in S, add 2x+s.
    #    - For each (s1, s2) in combinations_with_replacement(S, 2), add x+s1+s2.
    #    - If len(new_sums) != len(set(new_sums)), there's a collision among new sums. Reject.
    #    - If new_sums & T is not empty, there's a collision with existing sums. Reject.
    #    - If valid, accept.
    #    
    #    Complexity:
    #    - |S| = k.
    #    - Number of new sums: 1 + k + k(k+1)/2 = 1 + k + (k^2+k)/2 = (k^2+3k+2)/2 = (k+1)(k+2)/2.
    #    - For k=40, this is 41*42/2 = 861.
    #    - Checking against T (size ~11480) is O(861) using sets.
    #    - Checking for internal collisions is O(861) using sets.
    #    - Total per candidate: ~1000 ops.
    #    - Number of candidates: 4096 - 7 = 4089.
    #    - Total ops: 4089 * 1000 ~ 4 million. Very fast.
    #    
    #    So we can try all candidates.
    #    
    #    Should we try candidates in a specific order?
    #    - Ascending: Might fill small gaps first.
    #    - Descending: Might fill large gaps first.
    #    - Random: Might avoid local minima.
    #    
    #    The directive says "insert smaller integers into the gaps". This implies ascending order or focusing on small numbers.
    #    Let's try ascending order first. If it yields < 40, we can try other orders or randomization.
    #    
    #    However, to be safe and maximize the chance of hitting the theoretical max (40), we can try multiple strategies and pick the best result.
    #    Strategies:
    #    1. Skeleton + Ascending candidates.
    #    2. Skeleton + Descending candidates.
    #    3. Skeleton + Random candidates (multiple trials).
    #    4. No skeleton, just Random candidates (multiple trials).
    #    
    #    Given the time limit (60s), we can afford to run a few random trials.
    #    
    #    Let's implement:
    #    - A function `try_construct(candidates_order, skeleton)` that attempts to build the set.
    #    - Run it with different orders.
    #    - Return the best set found.
    #    
    #    Wait, the skeleton itself is a valid set. If we fail to add any, we return the skeleton (size 7).
    #    But the seed likely returns > 7.
    #    We need to ensure we beat the seed.
    #    
    #    Let's also consider that the skeleton might not be the best starting point.
    #    Maybe a different geometric progression?
    #    Base 4 is optimal for B_3.
    #    
    #    Let's code this up.

    def check_and_add(S, T, x):
        """
        Check if x can be added to S without creating duplicate triple sums.
        S: list of current elements.
        T: set of current triple sums.
        x: candidate element.
        Returns: (bool, new_T) where new_T is the updated set of sums if valid, else None.
        """
        new_sums = set()
        
        # 1. x, x, x
        new_sums.add(3 * x)
        
        # 2. x, x, s
        for s in S:
            new_sums.add(2 * x + s)
            
        # 3. x, s1, s2
        # Unordered pairs (s1, s2) with replacement from S
        # We can iterate over indices to handle replacement correctly
        k = len(S)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                new_sums.add(x + s1 + s2)
                
        # Check for internal collisions
        if len(new_sums) != len(new_sums): # This is always true, need to check set size vs list size?
            # Actually, we built a set, so internal collisions are already merged.
            # We need to know if there were collisions.
            # So we should build a list first, then check.
            pass
            
        # Let's rebuild with a list to check for collisions
        new_sums_list = []
        new_sums_list.append(3 * x)
        for s in S:
            new_sums_list.append(2 * x + s)
        for i in range(len(S)):
            s1 = S[i]
            for j in range(i, len(S)):
                s2 = S[j]
                new_sums_list.append(x + s1 + s2)
                
        if len(new_sums_list) != len(set(new_sums_list)):
            return False, None
            
        new_sums_set = set(new_sums_list)
        
        # Check collision with existing sums
        if new_sums_set & T:
            return False, None
            
        # Valid
        new_T = T.union(new_sums_set)
        return True, new_T

    def build_set(skeleton, candidates):
        S = list(skeleton)
        # Compute initial sums
        T = set()
        for t in combinations_with_replacement(S, 3):
            T.add(sum(t))
            
        for x in candidates:
            if x in S:
                continue
            valid, new_T = check_and_add(S, T, x)
            if valid:
                S.append(x)
                T = new_T
                if len(S) >= 40: # Theoretical max
                    break
                    
        return S

    # Define skeletons
    # Skeleton 1: Powers of 4
    skeleton1 = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Skeleton 2: Scaled powers of 4? No, must be <= 4096.
    # Maybe start with just the large ones?
    # Let's try a few variations.
    
    # Candidates: All numbers 1..4096
    all_candidates = list(range(1, 4097))
    
    # Strategy 1: Skeleton1 + Ascending
    cands_asc = [x for x in all_candidates if x not in skeleton1]
    set1 = build_set(skeleton1, cands_asc)
    
    # Strategy 2: Skeleton1 + Descending
    cands_desc = [x for x in reversed(all_candidates) if x not in skeleton1]
    set2 = build_set(skeleton1, cands_desc)
    
    # Strategy 3: No Skeleton + Ascending (Seed-like but with better check? No, seed is same)
    # Actually, let's try a random start.
    # Strategy 4: Random Skeleton?
    # Let's just run a few random trials with no skeleton, or with a small random skeleton.
    
    best_set = max([set1, set2], key=len)
    
    # Run a few random trials
    random.seed(42)
    for _ in range(10):
        # Random candidate order
        cands_rand = all_candidates[:]
        random.shuffle(cands_rand)
        # Start with empty set? Or small skeleton?
        # Let's start with empty set for variety.
        s_rand = build_set([], cands_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
        # Also try with skeleton1
        cands_rand2 = [x for x in cands_rand if x not in skeleton1]
        s_rand2 = build_set(skeleton1, cands_rand2)
        if len(s_rand2) > len(best_set):
            best_set = s_rand2

    # Ensure distinct and sorted? The task doesn't require sorted, but distinct is required.
    # Our construction ensures distinct.
    
    return best_set
# EVOLVE-BLOCK-END
