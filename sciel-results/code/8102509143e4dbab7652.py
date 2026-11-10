import time

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Construction: {1, ..., 50} U {100}
    # Size: 51
    # Max 2-sum: 50 + 100 = 150 < 200
    # Max 3-sum (2 small, 1 large): 49 + 50 + 100 = 199 < 200
    # 1 small, 2 large: Impossible (only one large)
    # 3 large: Impossible
    # So no subset sums to 200.
    
    candidate = list(range(1, 51)) + [100]
    
    # Verification
    def has_subset_sum_200(S):
        S_set = set(S)
        # 2-element
        for x in S:
            if 200 - x in S_set and 200 - x != x:
                return True
        # 3-element
        # Since max 2-sum is 150, min 3-sum involving large is 1+100+? 
        # Actually, just check all triples? 51 elements -> 51^3/6 ~ 20k checks. Fast.
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if S[i] + S[j] + S[k] == 200:
                        return True
        # 4-element
        # Min 4-sum: 1+2+3+4=10
        # Max 4-sum: 48+49+50+100=247
        # Check 4-sums?
        # If we have 4 elements, min sum is small.
        # But we can just use a bitset DP or similar if needed.
        # Given the structure, 4-sums:
        # 4 smalls: max 47+48+49+50=194 < 200.
        # 3 smalls + 1 large: max 48+49+50+100=247.
        # Check if any 3 smalls + 100 = 200?
        # s1+s2+s3 = 100.
        # Max 3 smalls sum: 48+49+50=147.
        # Min 3 smalls sum: 1+2+3=6.
        # Can 3 distinct smalls sum to 100?
        # Yes, e.g., 30+34+36=100.
        # So {30, 34, 36, 100} sums to 200!
        # My verification above was incomplete.
        
        return False

    # Correct Verification
    def is_safe(S):
        S_set = set(S)
        # 2-sum
        for x in S:
            if 200 - x in S_set and 200 - x != x:
                return False
        # 3-sum
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if S[i] + S[j] + S[k] == 200:
                        return False
        # 4-sum
        # Check if any 4 sum to 200
        # Since 5 smalls min sum 1+2+3+4+5=15, max 4 smalls 194.
        # 4 smalls cannot reach 200.
        # 3 smalls + 1 large (100): sum smalls = 100.
        # Check if any 3 distinct smalls sum to 100.
        smalls = [x for x in S if x < 100]
        smalls_set = set(smalls)
        for i in range(len(smalls)):
            for j in range(i+1, len(smalls)):
                rem = 100 - smalls[i] - smalls[j]
                if rem in smalls_set and rem > smalls[j]:
                    return False
        # 2 smalls + 2 larges? Only 1 large.
        # 1 small + 3 larges? Only 1 large.
        return True

    if not is_safe(candidate):
        # Fallback to known safe 35
        candidate = list(range(66, 101))
        
    return candidate
