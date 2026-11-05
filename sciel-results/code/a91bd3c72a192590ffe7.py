import time

def construct():
    start_time = time.time()
    time_limit = 85.0

    def is_valid_tail(k):
        """Check if [k, 256] is valid."""
        # Smallest is k.
        # Check a=k: pairs b,c in [k+1, 256] with b<=c, k !| (b+c)
        # Check a > k: pairs b,c in [a+1, 256] with b<=c, a !| (b+c)
        
        # Optimization: 
        # For a given a, we check if there exist b,c in [a+1, 256] such that (b+c) % a == 0.
        # This is equivalent to checking if the set of residues of [a+1, 256] mod a 
        # contains a pair summing to 0 mod a.
        
        for a in range(k, 257):
            # Elements greater than a are [a+1, 256]
            # If a+1 > 256, no pairs, continue.
            if a + 1 > 256:
                continue
            
            # We need to check if there exist b, c in [a+1, 256] with b<=c 
            # such that (b+c) % a == 0.
            
            # Let's collect residues of [a+1, 256] modulo a.
            # Note: The range [a+1, 256] has size 256-a.
            # If 256-a is large, we might have all residues.
            
            # Simple check:
            # Iterate b from a+1 to 256.
            # For each b, check if there is c in [b, 256] such that (b+c) % a == 0.
            # c % a = (-b) % a.
            # So we need to know if there is a c in [b, 256] with c % a == (-b) % a.
            
            # Precompute counts of residues for [a+1, 256]?
            # Or just iterate.
            # Since a is small (close to k), the range is large.
            # But 256 is small.
            
            # Let's use a set of residues seen so far in the loop for c?
            # No, b and c are both from the range.
            
            # Efficient way:
            # Let residues = set()
            # For c from 256 down to a+1:
            #   If (-b) % a in residues ... wait, b is varying too.
            
            # Let's just do O(N^2) for each a? No, N=256.
            # For each a, O((256-a)^2). Sum over a is O(N^3) ~ 16M.
            # This is fine for a single check.
            # But we call this in a loop?
            
            # Let's optimize the inner check.
            # For fixed a, let R be the set of residues of numbers in [a+1, 256] mod a.
            # We need to know if there exist x, y in [a+1, 256] (x<=y) such that (x+y)%a==0.
            
            # If the size of [a+1, 256] is >= a, then by Pigeonhole, all residues might be present?
            # Not necessarily.
            
            # Let's just iterate b and c.
            # To speed up, we can break early.
            
            found = False
            # We can precompute the set of residues available in [a+1, 256]
            # But we need x <= y.
            
            # Let's try a different approach for is_valid_tail.
            # For a from k to 256:
            #   Let M = 256 - a
            #   If M == 0: continue
            #   If M == 1: no pairs (b<=c requires at least 2 elements? No, b=c is allowed. So M>=1 allows b=c).
            #   Wait, b,c in [a+1, 256]. If M=1, only one element, say x. Pair (x,x). Check 2x % a.
            
            # Let's implement the check.
            # For efficiency, we can check if 2*b % a == 0 for any b, or if there are two distinct residues summing to 0.
            
            # Let's count residues in [a+1, 256]
            # count[r] = number of x in [a+1, 256] with x%a == r
            
            # If count[0] >= 2, then we have two multiples of a. Sum is 0 mod a. Invalid.
            # If count[0] == 1, we have one multiple. We need another multiple? No, sum of two multiples is 0.
            # If count[0] >= 1, we can form (x, x) if count[0]>=1? No, x is multiple, 2x is 0 mod a.
            # So if there is any multiple of a in [a+1, 256], then b=c=that multiple gives sum 2*mult, which is 0 mod a.
            # So if any x in [a+1, 256] is divisible by a, invalid.
            
            # Otherwise, no element is divisible by a.
            # Then we need to check if there exist x, y (distinct or same? if same, 2x%a!=0 since x%a!=0) 
            # such that (x+y)%a==0.
            # Since x%a != 0, we need x%a + y%a == a.
            # So we need to check if there exist residues r, s such that r+s=a and count[r]>=1, count[s]>=1.
            # If r != s, we need count[r]>=1 and count[s]>=1.
            # If r == s (i.e., 2r=a), we need count[r]>=2.
            
            # This is much faster!
            
            # Step 1: Check if any x in [a+1, 256] is divisible by a.
            # The smallest multiple of a greater than a is 2a.
            # So if 2a <= 256, then 2a is in the range and is divisible by a.
            # So if 2a <= 256, it is INVALID.
            
            if 2 * a <= 256:
                return False
            
            # Step 2: If 2a > 256, then no element in [a+1, 256] is divisible by a.
            # We need to check residues.
            # Range is [a+1, 256]. Size M = 256 - a.
            # Since 2a > 256, a > 128.
            # So a is in [129, 256].
            # M = 256 - a < 128.
            
            # We need to check if there exist x, y in [a+1, 256] such that (x+y)%a==0.
            # Since x,y < 2a (as x,y <= 256 < 2a), x+y < 4a.
            # x+y can be a, 2a, 3a.
            # But x,y > a, so x+y > 2a.
            # So x+y must be 3a.
            # So we need to check if there exist x, y in [a+1, 256] such that x+y = 3a.
            
            # This is a subset sum problem for 2 elements.
            # We can check this in O(M).
            # Let's create a set of elements in [a+1, 256].
            # For each x in [a+1, 256], check if 3a - x is in the set and >= x (to ensure b<=c, though order doesn't matter for existence).
            
            elems = set(range(a + 1, 257))
            for x in elems:
                y = 3 * a - x
                if y in elems:
                    # We found a pair.
                    # Check if y >= x? Not required for validity check, just existence.
                    # But we must ensure y is in range [a+1, 256].
                    # Since x in [a+1, 256], y = 3a - x.
                    # Min y (when x max) = 3a - 256.
                    # Max y (when x min) = 3a - (a+1) = 2a - 1.
                    # We need y >= a+1 and y <= 256.
                    if y >= a + 1 and y <= 256:
                        return False
            
        return True

    # Find smallest k such that [k, 256] is valid
    # We know if k is valid, k+1 is valid.
    # So we can binary search or just iterate from 129 upwards?
    # We know 2a > 256 is required for validity of a. So a > 128.
    # So k must be at least 129?
    # If k=128, a=128, 2a=256 <= 256. Invalid.
    # So k >= 129.
    
    best_set = []
    best_size = 0
    
    # Try k from 129 to 256
    for k in range(129, 257):
        if time.time() - start_time > time_limit - 5:
            break
        if is_valid_tail(k):
            S = list(range(k, 257))
            # Try to add smaller elements
            for x in range(k - 1, 0, -1):
                if time.time() - start_time > time_limit - 5:
                    break
                # Check if x can be added to S
                # x is the new smallest.
                # Condition: for all b, c in S with x < b <= c, x !| (b+c).
                # S is [k, 256].
                # We need to check if any pair b,c in [k, 256] sums to multiple of x.
                
                # Similar logic to is_valid_tail but for specific x and range [k, 256].
                
                # 1. If 2x <= 256 and 2x >= k, then 2x is in range?
                # No, 2x is a sum. We need b,c such that b+c is multiple of x.
                # Smallest sum is 2k. Largest is 512.
                
                # Let's check if there exist b, c in [k, 256] with b<=c such that (b+c) % x == 0.
                
                # Case 1: b=c. 2b % x == 0.
                # Case 2: b!=c. b+c = m*x.
                
                # Efficient check:
                # If 2k > 256? No, k>=129, 2k >= 258.
                # So b+c >= 258.
                # Multiples of x in [258, 512].
                
                # Let's just iterate b in [k, 256].
                # For each b, check if there is c in [b, 256] such that (b+c) % x == 0.
                # c % x = (-b) % x.
                # We need c in [b, 256] with c % x == (-b) % x.
                
                # Precompute residues of [k, 256] mod x?
                # Or just check.
                
                valid_add = True
                # Optimization: if x is small, many multiples.
                # Let's check 2b first.
                # If there exists b in [k, 256] such that 2b % x == 0, invalid.
                # 2b = m*x. b = m*x/2.
                # So if x/2 is integer, and m*x/2 in [k, 256], invalid.
                # If x is odd, 2b % x == 0 implies b % x == 0.
                # So if there is a multiple of x in [k, 256], and x is odd, invalid?
                # No, 2b % x == 0. If x odd, gcd(2,x)=1, so b % x == 0.
                # So if x is odd and there is a multiple of x in [k, 256], then 2*(that multiple) is 0 mod x.
                
                # Let's just do the pair check.
                # To speed up, we can break early.
                
                # Let's use a set of residues seen?
                # No, b and c are from the same set.
                
                # Iterate b from k to 256.
                # For each b, target_res = (-b) % x.
                # We need to know if there is a c in [b, 256] with c % x == target_res.
                
                # We can precompute for each residue r, the list of c in [k, 256] with c%x==r.
                # Then for each b, we check if any c in that list is >= b.
                
                # Since x < k (we are adding x < k), x is at most 128.
                # Range [k, 256] size is 256-k+1 <= 128.
                # So this is fast.
                
                # Build residue lists
                res_lists = {r: [] for r in range(x)}
                for c in range(k, 257):
                    res_lists[c % x].append(c)
                
                # Sort each list? They are already sorted.
                
                for b in range(k, 257):
                    target = (-b) % x
                    # Check if there is c in res_lists[target] such that c >= b
                    # Since res_lists[target] is sorted, we can binary search or just check max.
                    # If max(res_lists[target]) >= b, then there is such a c.
                    # Wait, we need c >= b.
                    # If the largest element in res_lists[target] is >= b, it works.
                    # But we also need c <= 256, which is true.
                    
                    if res_lists[target]:
                        if res_lists[target][-1] >= b:
                            valid_add = False
                            break
                
                if valid_add:
                    S.append(x)
                    S.sort()
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                
            # Since smaller k gives larger tail, and we iterate k increasing, 
            # the first valid k gives the largest tail.
            # But adding smaller elements might make a larger k (smaller tail) better?
            # Unlikely, but we check all k.
            
    return best_set
