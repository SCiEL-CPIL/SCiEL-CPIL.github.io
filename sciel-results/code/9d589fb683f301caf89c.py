"""Program: large 3-sum-free-pair-sum set in 1..128 via structured greedy."""
# EVOLVE-BLOCK-START
import random

N = 128

def construct():
    # Strategy:
    # 1. Start with S_large = {65..128}
    # 2. Add elements from {1..64} if they don't create a forbidden triple.
    # A triple {a,b,c} is forbidden if a+b, a+c, b+c are all in S.
    # If a,b are in S_large, a+b > 128, so safe.
    # If a in S_large, b,c in S_small:
    #   Sums: a+b, a+c, b+c.
    #   b+c is small. a+b, a+c are large.
    #   Forbidden if b+c in S and a+b in S and a+c in S.
    #   If S_small is sum-free, b+c not in S_small.
    #   If b+c < 65, b+c not in S_large. So b+c not in S. Safe.
    #   If b+c >= 65, b+c in S_large. Then we need a+b in S and a+c in S.
    #   This is likely. So we should ensure b+c < 65 for all b,c in S_small.
    #   This implies S_small must be a set where any two distinct elements sum to < 65.
    #   And S_small must be sum-free (so b+c not in S_small).
    
    # Let's construct S_small greedily from 1..64.
    # Constraint: for any a,b in S_small (distinct), a+b < 65 and a+b not in S_small.
    
    S = set(range(65, 129))
    
    # Candidates from 1 to 64
    # To maximize size, we want to pick as many as possible.
    # Picking smaller numbers allows more combinations?
    # If we pick 1, we can pick up to 63 (1+63=64).
    # If we pick 32, we can pick up to 32 (32+32=64, but distinct so 31).
    
    # Let's try to pick a large sum-free set with max sum < 65.
    # This is equivalent to picking a subset of {1..64} such that:
    # 1. No a+b in set (for distinct a,b)
    # 2. max(a,b) + second_max(a,b) < 65
    
    # Greedy: iterate 1..64, add if valid.
    # To be fast, we can just hardcode a good set or use a quick greedy.
    
    S_small = set()
    for x in range(1, 65):
        # Check if x can be added
        ok = True
        for y in S_small:
            if x == y: continue
            s = x + y
            if s >= 65:
                ok = False
                break
            if s in S_small:
                ok = False
                break
        if ok:
            S_small.add(x)
            
    S.update(S_small)
    
    # This greedy might be suboptimal. 
    # Let's try a few random restarts on the small part to see if we can get more.
    # But the structure is rigid. 
    # Actually, the set {1..32} is not sum-free.
    # The set {17..32} is sum-free? 17+18=35 (in set). No.
    # The set of odds {1,3,..,63} is sum-free. But sums are large.
    # We need sums < 65.
    # So we are limited to small numbers.
    
    # Let's verify the size of the greedy S_small.
    # 1: ok. S={1}
    # 2: 1+2=3<65, 3 not in S. ok. S={1,2}
    # 3: 1+3=4, 2+3=5. ok. S={1,2,3}
    # ...
    # This will pick 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32?
    # Check 32: 31+32=63 < 65. 31 in S. 63 not in S. ok.
    # Check 33: 32+33=65. Not < 65. Fail.
    # So S_small = {1..32}. Size 32.
    # Is {1..32} sum-free? No. 1+2=3 in S.
    # My check `if s in S_small` should have caught this.
    # When adding 3: 1+2=3. 3 is the candidate. `s in S_small` checks if sum is in CURRENT S_small.
    # Current S_small is {1,2}. 3 is not in {1,2}. So it adds 3.
    # But then {1,2,3} is in S. 1+2=3 in S.
    # The condition for validity is that a+b, a+c, b+c are in S.
    # If S_small = {1,2,3}, and we add c=65 (large).
    # 1+2=3 in S. 1+65=66 in S. 2+65=67 in S.
    # So {1,2,65} is bad.
    # So my greedy logic `if s in S_small` is insufficient.
    # I need to ensure that if I add x, I don't create a bad triple with existing elements.
    
    # Correct Check for adding x to S:
    # For all pairs (a,b) in S (a<b):
    #   If x+a in S and x+b in S and a+b in S -> Bad.
    
    # This is O(N^2) per addition. N=128. 128*128*128 ~ 2M ops. Fast enough.
    
    def can_add(x, S_set):
        elems = list(S_set)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            xa = x + a
            if xa > 128: continue
            if xa not in S_set: continue
            for j in range(i+1, n):
                b = elems[j]
                xb = x + b
                if xb > 128: continue
                if xb not in S_set: continue
                ab = a + b
                if ab == x or ab in S_set:
                    return False
        return True

    # Restart with empty set and greedy add?
    # Or start with large set and add smalls?
    
    best = None
    best_len = 0
    rng = random.Random(42)
    
    for trial in range(100):
        # Random order of 1..128
        perm = list(range(1, 129))
        rng.shuffle(perm)
        
        S_set = set()
        for x in perm:
            if can_add(x, S_set):
                S_set.add(x)
                
        if len(S_set) > best_len:
            best_len = len(S_set)
            best = list(S_set)
            
    return best
# EVOLVE-BLOCK-END
