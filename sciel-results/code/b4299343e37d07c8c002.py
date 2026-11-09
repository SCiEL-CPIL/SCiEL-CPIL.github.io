"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # We want to maximize the size of the set.
    # The constraint is: for all non-empty subsets A, B, sum(A) does not divide sum(B) 
    # unless A=B. Also, all subset sums must be distinct (implied by the divide check 
    # if we consider equal sums: if sum(A)==sum(B) and A!=B, then sum(A) divides sum(B), 
    # but the prompt says "no two distinct resulting values may divide one another" 
    # AND "Equal sum values are counted once... Two different subsets with the same sum 
    # still make the set invalid").
    # Actually, the prompt clarifies: "no two distinct resulting values may divide one another".
    # It also says: "Two different subsets with the same sum still make the set invalid".
    # So the condition is:
    # 1. All subset sums are distinct.
    # 2. For any two distinct subset sums S1, S2, S1 does not divide S2 and S2 does not divide S1.
    
    # Let's verify the logic:
    # If we have a set X, and we compute all subset sums.
    # If sum(A) == sum(B) for A != B, invalid.
    # If sum(A) divides sum(B) for A != B, invalid.
    # Note: If A is a subset of B, sum(A) < sum(B) (assuming positive ints).
    # If sum(A) divides sum(B), it's invalid.
    
    # Strategy: Greedy construction.
    # Start with an empty set.
    # Iterate through candidate integers from 256 down to 1.
    # For each candidate x, check if adding x to the current set preserves the validity.
    # To check validity efficiently:
    # Let S be the set of current subset sums.
    # New subset sums will be S ∪ {s + x for s in S} ∪ {x}. (Actually S ∪ (S+x)).
    # Wait, the new set of subset sums is S_new = S ∪ {s+x | s in S}.
    # Note that the empty sum 0 is usually excluded from "non-empty subset sums", 
    # but the problem says "Form every NONEMPTY subset sum".
    # So S does not contain 0.
    # However, when adding x, the new sums are {x} ∪ {s+x | s in S}.
    # Let's maintain a set of valid subset sums.
    
    current_sums = set()
    chosen = []
    
    # Optimization: 
    # If we have k elements, there are 2^k - 1 subset sums.
    # Max size is 30. 2^30 is ~10^9, which is too large to store in a set in Python quickly 
    # if we do this for every candidate.
    # However, we are going from 256 down to 1.
    # If we pick large numbers, the sums grow quickly.
    # Let's estimate the max possible size.
    # If we pick numbers that are powers of 2, e.g., 1, 2, 4, ..., 2^27.
    # Subset sums are all distinct integers from 1 to 2^28-1.
    # Do they divide each other?
    # 1 divides everything. So 1 cannot be in the set if any other sum exists?
    # Wait. "No two distinct resulting values may divide one another."
    # If 1 is a subset sum, and S is another subset sum, 1 divides S. 
    # So 1 cannot be a subset sum unless it's the ONLY subset sum.
    # This implies we cannot have 1 in our set of integers, because {1} is a subset sum.
    # If we have {2}, sums are {2}. Valid.
    # If we have {2, 4}, sums are {2, 4, 6}. 
    # 2 divides 4? Yes. Invalid.
    # So {2, 4} is invalid.
    
    # This problem is quite restrictive.
    # Let's look at the seed: [31, 32, 34].
    # Sums: 31, 32, 34, 31+32=63, 31+34=65, 32+34=66, 31+32+34=97.
    # Set: {31, 32, 34, 63, 65, 66, 97}
    # Check divisibility:
    # 31 divides 63? No. 31 divides 65? No. 31 divides 66? No. 31 divides 97? No.
    # 32 divides 63? No. 32 divides 65? No. 32 divides 66? No. 32 divides 97? No.
    # 34 divides 63? No. 34 divides 65? No. 34 divides 66? No. 34 divides 97? No.
    # 63 divides 65? No. 63 divides 66? No. 63 divides 97? No.
    # 65 divides 66? No. 65 divides 97? No.
    # 66 divides 97? No.
    # Also check reverse (larger divides smaller? No, only if equal, but they are distinct).
    # So [31, 32, 34] is valid. Size 3.
    
    # Can we do better?
    # The constraint "1 divides everything" means 1 cannot be a sum.
    # So the smallest element must be > 1? No, if the set is {2}, sum is 2.
    # If set is {3}, sum is 3.
    # If set is {2, 5}, sums: 2, 5, 7.
    # 2 divides 5? No. 2 divides 7? No. 5 divides 2? No. 5 divides 7? No.
    # So {2, 5} is valid. Size 2.
    # {2, 5, 11}? Sums: 2, 5, 11, 7, 13, 16, 18.
    # 2 divides 16? Yes. Invalid.
    # {3, 5, 11}? Sums: 3, 5, 11, 8, 14, 16, 19.
    # 3 divides 14? No. 3 divides 16? No.
    # 5 divides 14? No. 5 divides 16? No.
    # 8 divides 16? Yes. Invalid.
    
    # It seems hard to get large sets.
    # Let's try a greedy approach with backtracking if it's fast enough, 
    # or just greedy if it finds a good solution.
    
    # Given the 60s limit, and the exponential growth of subset sums, 
    # a greedy approach trying to add the largest possible numbers might work well.
    # Why largest? Larger numbers are less likely to divide each other.
    # Also, if a divides b, b must be at least 2a.
    # So if we pick numbers close to 256, their sums will be large.
    
    # Let's implement a function to check if a candidate x can be added to current_sums.
    
    def can_add(x, current_sums):
        # New sums are current_sums ∪ {s+x for s in current_sums} ∪ {x}
        # Actually, the new set of sums is:
        # new_sums = set(current_sums)
        # new_sums.add(x)
        # for s in current_sums:
        #     new_sums.add(s+x)
        
        # We need to check two conditions on new_sums:
        # 1. All elements are distinct (guaranteed by using a set, but we must ensure 
        #    that s+x != s' for s,s' in current_sums, and s+x != x, etc. 
        #    If s+x = s', then x = s'-s. Since x > 0 and s, s' are sums of positive ints, 
        #    this is possible. E.g. current sums {2, 5}, x=3. 2+3=5. Duplicate! 
        #    So we must check for duplicates first.)
        
        # 2. No two distinct elements divide each other.
        
        # Efficient check:
        # Generate the new sums.
        new_sums_list = list(current_sums)
        new_sums_list.append(x)
        for s in current_sums:
            new_sums_list.append(s + x)
            
        # Check for duplicates
        if len(set(new_sums_list)) != len(new_sums_list):
            return False, set()
            
        new_sums = set(new_sums_list)
        
        # Check divisibility
        # Sort the sums to make it easier? 
        # For each a in new_sums, for each b in new_sums where b > a, check if b % a == 0.
        # This is O(N^2) where N is number of sums.
        # N grows exponentially. For size 10, N=1023. 1023^2 ~ 10^6. 
        # For size 15, N=32767. 32767^2 ~ 10^9. Too slow in Python.
        
        # We need a smarter check or smaller N.
        # If N is small (size < 10), O(N^2) is fine.
        # If we aim for size > 10, we need optimization.
        
        # Optimization: 
        # If a divides b, then b = k*a.
        # We can iterate through sorted sums.
        sorted_sums = sorted(new_sums)
        for i in range(len(sorted_sums)):
            a = sorted_sums[i]
            # Check multiples of a
            # b must be in new_sums.
            # Instead of checking all b, we can check if any multiple of a is in new_sums.
            # But new_sums is a set, so `if (a*k) in new_sums` is O(1).
            # How many multiples to check?
            # Max sum is roughly sum(chosen) + x.
            # Max sum < 30 * 256 = 7680.
            # So for each a, we check k from 2 to 7680//a.
            # This is much faster than O(N^2) if a is large.
            
            # But if a is small (e.g., 2), we check up to 3840 multiples.
            # If we have many small sums, it might be slow.
            # However, if a is small, it's likely to divide something.
            
            # Let's try this approach.
            max_sum = sorted_sums[-1]
            for k in range(2, max_sum // a + 1):
                if (a * k) in new_sums:
                    return False, set()
                    
        return True, new_sums

    # Greedy: try to add numbers from 256 down to 1.
    # We want to maximize the count.
    # Starting with large numbers might be better because their sums are large 
    # and less likely to divide each other?
    # Actually, if we start with small numbers, we get more sums quickly, 
    # which makes it harder to add more.
    # So starting with LARGE numbers is likely better for maximizing count?
    # Or maybe starting with small numbers allows more "slots"?
    # Let's test both or just one. 
    # Given the seed [31, 32, 34] are mid-range.
    
    # Let's try descending order (256, 255, ...)
    
    current_sums = set()
    chosen = []
    
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        ok, new_sums = can_add(x, current_sums)
        if ok:
            chosen.append(x)
            current_sums = new_sums
            # If we reach a certain size, maybe we can stop? 
            # Max size is 30. 
            # But 2^30 is huge. 
            # If we get size 10, that's 1023 sums. 
            # If we get size 15, that's 32k sums. 
            # The check `can_add` will become slower.
            
    # The greedy approach might get stuck at a small size.
    # Is there a known maximum?
    # Let's just return what we found.
    
    # If the greedy finds very few, maybe we should try a different order?
    # But we have a time limit.
    # Let's assume descending is good.
    
    # One issue: `can_add` creates a new list and set every time.
    # For large current_sums, this is expensive.
    
    # Let's optimize `can_add` slightly by passing the set and reusing logic.
    # Actually, the current implementation is okay for small sizes.
    # If size gets to 12, 4095 sums. 4095 * (max_sum/avg_a) checks.
    # Max sum ~ 3000. Avg a ~ 1500. ~ 2 checks per a. 4095 * 2 = 8000 checks.
    # This is fast.
    
    # If size gets to 15, 32767 sums. 32767 * 2 = 65k checks.
    # Still fast in Python? 65k iterations is < 0.1s.
    # So we can likely go up to size 15-16.
    
    return chosen

# EVOLVE-BLOCK-END
