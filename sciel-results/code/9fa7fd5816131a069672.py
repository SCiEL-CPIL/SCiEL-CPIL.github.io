import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # The problem asks for distinct positive integers <= 4096.
    # Maximize cardinality.
    
    def check_addition(S, x, sums_set):
        """
        Check if x can be added to S while maintaining unique triple sums.
        S is a sorted list of current elements.
        sums_set is a set of all current triple sums.
        Returns (True, new_sums_list) if valid, (False, None) otherwise.
        """
        n = len(S)
        new_sums = set()
        
        # 1. (x, x, x)
        s1 = 3 * x
        if s1 in sums_set or s1 in new_sums:
            return False, None
        new_sums.add(s1)
        
        # 2. (x, x, s) for s in S
        for s in S:
            s2 = 2 * x + s
            if s2 in sums_set or s2 in new_sums:
                return False, None
            new_sums.add(s2)
            
        # 3. (x, s, t) for s <= t in S
        for i in range(n):
            s = S[i]
            for j in range(i, n):
                t = S[j]
                s3 = x + s + t
                if s3 in sums_set or s3 in new_sums:
                    return False, None
                new_sums.add(s3)
                
        return True, list(new_sums)

    def greedy_search(order):
        """
        Greedily add elements from 'order' to the set.
        Returns (set_of_elements, set_of_sums).
        """
        S = []
        sums_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = check_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
                S.sort() # Keep sorted for consistency, though not strictly needed for logic
                # Optimization: If we hit a limit or time, break
                if len(S) >= 100: # Upper bound check, task says 1..80, but let's see
                    break
                    
        return S, sums_set

    def local_search(S, sums_set):
        """
        Try to improve the set by swapping elements.
        We try to replace an element s_i with a new candidate x not in S.
        This is complex to verify efficiently from scratch.
        Alternative: Try to add more elements if there's space? No, S is maximal for the given order.
        
        Better Local Search: 
        1. Identify "expensive" elements (large values) that block many candidates.
        2. Try to remove a large element and see if we can add two smaller ones? 
           Or just try to add elements from a shuffled list of remaining candidates.
        
        Given the time constraints, a simple randomized restart greedy is often more effective than complex local search for this specific B_h problem in short time.
        """
        pass

    best_S = []
    best_sums = set()
    
    # Strategy 1: Standard Greedy (1, 2, 3...)
    # This is the baseline.
    S1, sums1 = greedy_search(range(1, 4097))
    if len(S1) > len(best_S):
        best_S = S1
        best_sums = sums1
        
    # Strategy 2: Randomized Greedy
    # We want to find a set that might allow more elements.
    # Sometimes picking smaller numbers greedily blocks larger numbers that could fit.
    # Or vice versa.
    
    num_restarts = 100 # Will be limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random permutation of candidates
        # We don't need all 4096, but a good subset.
        # Actually, the greedy check is O(N^2). 
        # If we check 4096 candidates, and S grows to 80, cost is ~ 4096 * 80^2 = 26M ops.
        # This is fast in Python? Maybe 1-2 seconds per full pass if optimized.
        # Let's try to be smart.
        
        # Random shuffle of a range
        # To maximize cardinality, we generally want to pack small numbers?
        # B_3 sequences usually grow exponentially. 
        # A known B_3 sequence starts: 1, 2, 4, 8, 16... (powers of 2)
        # If we use powers of 2, triple sums are unique?
        # Sum of 3 powers of 2 is a number with at most 3 bits set (if distinct) or fewer.
        # Actually, if we use powers of 2, the sum is unique by binary representation?
        # Wait. 1+1+2 = 4. 1+2+2 = 5. 
        # If we use $2^i$, the sum is $\sum c_i 2^i$ where $c_i \in \{0,1,2,3\}$.
        # This is NOT unique in base 2. E.g., $1+1+1=3$ and $1+2=3$ (but we need 3 terms).
        # $1+1+1=3$. $1+1+1$ is the only way to get 3 from powers of 2 with 3 terms?
        # $1+1+2=4$. $1+1+2$ vs $1+1+2$.
        # Let's check: $2^0, 2^1, 2^2$.
        # $1+1+1=3$.
        # $1+1+2=4$.
        # $1+2+2=5$.
        # $2+2+2=6$.
        # $1+1+4=6$. Collision! $2+2+2=6$ and $1+1+4=6$.
        # So powers of 2 do NOT work for B_3.
        
        # B_3 sequences are related to base 3 or higher?
        # If we use $3^i$, sums are unique in base 3?
        # Digits in base 3 for sum of 3 terms are in $0..3$.
        # $1+1+1 = 10_3$.
        # $1+1+3 = 11_3$? No, $1+1+3=5=12_3$.
        # $1+3+3=7=21_3$.
        # $3+3+3=9=100_3$.
        # $1+1+9=11=102_3$.
        # It seems base 3 powers might work?
        # Let's check $1, 3, 9$.
        # $1+1+1=3$.
        # $1+1+3=5$.
        # $1+3+3=7$.
        # $3+3+3=9$.
        # $1+1+9=11$.
        # $1+3+9=13$.
        # $3+3+9=15$.
        # $9+9+9=27$.
        # $1+9+9=19$.
        # $3+9+9=21$.
        # Are there collisions?
        # $1+1+1=3$.
        # $3=3$. (Term 3 is in set).
        # Wait, the set is $\{1, 3, 9\}$.
        # Triple (1,1,1) sum=3.
        # Triple (3,0,0) not allowed, terms must be from set.
        # So 3 is a sum from (1,1,1).
        # Is 3 in the set? Yes.
        # Does that matter? The condition is "Every unordered triple... must have a different sum".
        # It does NOT say the sum cannot be in the set.
        # So $1+1+1=3$ is fine.
        # But $1+1+1=3$ and $3+3+3=9$.
        # What about $1+1+3=5$?
        # $1+3+3=7$?
        # $3+3+3=9$.
        # $1+1+9=11$.
        # $1+3+9=13$.
        # $3+3+9=15$.
        # $1+9+9=19$.
        # $3+9+9=21$.
        # $9+9+9=27$.
        
        # Let's check base 3 powers for collisions.
        # Sum is $\sum d_i 3^i$ with $d_i \in \{0,1,2,3\}$.
        # If $d_i=3$, it carries over to $1 \cdot 3^{i+1}$.
        # This causes ambiguity.
        # Example: $3^0 + 3^0 + 3^0 = 3 = 3^1$.
        # So $(1,1,1)$ sum is 3.
        # $(3,3,3)$ sum is 27.
        # $(1,1,3)$ sum is 5.
        # $(1,3,3)$ sum is 7.
        # $(3,3,3)$ sum is 27.
        # $(1,1,9)$ sum is 11.
        # $(1,3,9)$ sum is 13.
        # $(3,3,9)$ sum is 15.
        # $(9,9,9)$ sum is 27.
        # Collision! $(3,3,3)$ sum 27 and $(9,9,9)$ sum 27.
        # So base 3 powers do NOT work.
        
        # We need a larger base.
        # For B_h, base $h+1$? Or $2h+1$?
        # If we use base $B$, max digit is $h$.
        # If $B > h$, no carries.
        # For B_3, max digit is 3.
        # If base is 4, digits are 0,1,2,3. No carry.
        # So powers of 4 should work!
        # Let's verify.
        # Set: $1, 4, 16, 64, \dots$
        # Sum of 3 elements: $\sum c_i 4^i$ where $c_i \in \{0,1,2,3\}$.
        # Since max coeff is 3 < 4, the representation in base 4 is unique.
        # Thus, all sums are unique.
        
        # How many powers of 4 fit in 4096?
        # $4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024, 4^6=4096$.
        # So we can have $\{1, 4, 16, 64, 256, 1024, 4096\}$.
        # Cardinality = 7.
        
        # The seed program found 1 (or very few) because it was slow?
        # No, the seed program:
        # `for x in range(1, 513):`
        # It tries 1, 2, 3...
        # 1: OK.
        # 2: (1,1,1)=3, (1,1,2)=4, (1,2,2)=5, (2,2,2)=6. OK.
        # 3: (1,1,1)=3, (1,1,2)=4, (1,2,2)=5, (2,2,2)=6, (1,1,3)=5 -> Collision with (1,2,2).
        # So 3 is rejected.
        # 4: (1,1,4)=6 -> Collision with (2,2,2). Rejected.
        # 5: (1,1,5)=7. (1,2,5)=8. (2,2,5)=9. (1,5,5)=11. (2,5,5)=12. (5,5,5)=15.
        # Existing sums: 3,4,5,6.
        # New sums: 7,8,9,11,12,15. No collision.
        # So 5 is accepted.
        # Set: {1, 2, 5}.
        # Sums:
        # 1,1,1=3
        # 1,1,2=4
        # 1,2,2=5
        # 2,2,2=6
        # 1,1,5=7
        # 1,2,5=8
        # 2,2,5=9
        # 1,5,5=11
        # 2,5,5=12
        # 5,5,5=15
        
        # Next candidate 6:
        # 1,1,6=8 -> Collision with 1,2,5=8. Rejected.
        # 7:
        # 1,1,7=9 -> Collision with 2,2,5=9. Rejected.
        # 8:
        # 1,1,8=10.
        # 1,2,8=11 -> Collision with 1,5,5=11. Rejected.
        # 9:
        # 1,1,9=11 -> Collision.
        # 10:
        # 1,1,10=12 -> Collision with 2,5,5=12.
        # 11:
        # 1,1,11=13.
        # 1,2,11=14.
        # 2,2,11=15 -> Collision with 5,5,5=15.
        
        # It seems greedy with small numbers gets stuck quickly.
        # The "inspiration" score 1.78 suggests ~14 elements (if seed was 8).
        # Or maybe the seed was smaller.
        
        # The powers of 4 strategy gives 7 elements.
        # Can we do better?
        # We can mix.
        # The greedy approach with randomization might find a larger set.
        
        # Let's run the randomized greedy.
        
        # To speed up, we can limit the candidate pool.
        # But 4096 is small enough.
        
        # Optimization in check_addition:
        # We can break early.
        
        # Let's refine the greedy search to be faster.
        
        # We will perform multiple random restarts.
        
        # One specific strategy: 
        # Start with powers of 4, then try to add more?
        # Or just random greedy.
        
        # Let's stick to random greedy with time limit.
        
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Random shuffle of candidates
            # We can prioritize smaller numbers? Or random?
            # Random permutation of 1..4096 is expensive to generate and iterate.
            # Instead, pick a random subset or random order.
            
            # A good heuristic for B_h sequences is to pick numbers that are "far" from existing sums.
            # But random is simple.
            
            # To make it faster, we can just shuffle a list of candidates.
            # 4096 items. Shuffle is O(N).
            # Iterating and checking is O(N * |S|^2).
            
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
            
            S, sums_set = greedy_search(candidates)
            
            if len(S) > len(best_S):
                best_S = S
                best_sums = sums_set
                
            # Early stop if we find a very large set?
            # Theoretical max?
            # Sums range from 3 to 3*4096 = 12288.
            # Number of triples is C(|S|+2, 3).
            # We need C(|S|+2, 3) <= 12288.
            # |S|^3 / 6 <= 12288 => |S|^3 <= 73728 => |S| <= 42.
            # So max cardinality is around 40-42.
            
            # If we find 40, we can stop.
            if len(best_S) >= 40:
                break
                
    return best_S

# Quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
