"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random
import time

N = 128

def construct():
    best = None
    best_len = 0

    rng = random.Random(12345)

    def can_add(x, S_mask):
        # A_mask contains bits a such that a is in S and x+a is in S
        A_mask = S_mask & (S_mask >> x)
        
        # If fewer than 2 candidates in A, no pair can form a conflict
        if A_mask.bit_count() < 2:
            return True
        
        # Extract indices of set bits in A_mask
        a_list = []
        temp = A_mask
        while temp:
            lb = temp & -temp
            a_list.append(lb.bit_length() - 1)
            temp ^= lb
            
        # Check all pairs in a_list
        for i in range(len(a_list)):
            a = a_list[i]
            for j in range(i + 1, len(a_list)):
                b = a_list[j]
                s = a + b
                if s == x or ((S_mask >> s) & 1):
                    return False
        return True

    start_time = time.time()
    max_time = 55.0  # Leave 5s buffer

    # We will run as many trials as possible
    trial = 0
    while time.time() - start_time < max_time:
        # Random permutation
        perm = list(range(1, 129))
        rng.shuffle(perm)

        S_mask = 0
        S_count = 0

        for x in perm:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                S_count += 1

        if S_count > best_len:
            best_len = S_count
            best = S_mask
            # Optional: early stop if we hit a theoretical upper bound?
            # Upper bound is 128. Unlikely to reach.
            # But if we get 80+, it's good.
            
        trial += 1

    # Convert best mask to list
    if best is None:
        return list(range(65, 129))
        
    result = []
    temp = best
    while temp:
        lb = temp & -temp
        result.append(lb.bit_length() - 1)
        temp ^= lb
        
    # Verify (just in case)
    def is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True

    if not is_valid(result):
        return list(range(65, 129))

    return result
# EVOLVE-BLOCK-END
