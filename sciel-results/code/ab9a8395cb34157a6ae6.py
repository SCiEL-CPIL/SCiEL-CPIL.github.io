import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute suffix sums for max sum pruning?
    # suffix_sum[i] = sum of divisors[i:]
    # But we only need sum of first 'k' elements from i.
    # Given small N, we can just sum on the fly or precompute prefix sums?
    # Prefix sums on reversed list?
    # Let's just use a helper to get max sum of k items from index i.
    
    # To speed up, we can precompute sums of contiguous blocks?
    # Or just use the fact that divisors are sorted descending.
    # Max sum of k items from index i is sum(divisors[i:i+k]).
    
    def max_sum_from(i, k):
        if i + k > num_divs:
            return -1 # Not enough elements
        return sum(divisors[i:i+k])
        
    # Min sum of k items from index i?
    # The smallest k items in divisors[i:] are the last k items of the whole list?
    # No, they are the last k items of divisors[i:].
    # Since divisors are sorted descending, the smallest are at the end.
    # So min sum is sum(divisors[num_divs-k:num_divs]) if i <= num_divs-k.
    # If i > num_divs-k, then the available range is smaller.
    # Actually, if we pick k items from i..end, the minimum sum is the sum of the k smallest items in that range.
    # Since sorted desc, these are the last k items of the range i..end.
    # So min_sum_from(i, k) = sum(divisors[num_divs-k:num_divs]) if i <= num_divs-k?
    # Wait. The range is divisors[i:].
    # The k smallest items in this range are divisors[num_divs-k:num_divs] ONLY IF i <= num_divs-k.
    # If i > num_divs-k, then we don't have k items? No, we need to check if enough items exist.
    # If num_divs - i < k, return inf.
    # The k smallest items in divisors[i:] are divisors[num_divs-k:num_divs] if i <= num_divs-k.
    # If i > num_divs-k, then the k smallest are... wait.
    # Example: [10, 9, 8, 7, 6]. i=2 (val 8). k=2.
    # Range: [8, 7, 6]. Smallest 2: [7, 6]. Sum 13.
    # Formula: sum(divisors[5-2:5]) = sum(divisors[3:5]) = 7+6=13. Correct.
    # What if i=3 (val 7)? Range [7, 6]. k=2. Smallest 2: [7,6]. Sum 13.
    # Formula: sum(divisors[3:5]) = 13. Correct.
    # What if i=4 (val 6)? Range [6]. k=2. Not enough items.
    
    def min_sum_from(i, k):
        if num_divs - i < k:
            return float('inf')
        # The k smallest items in divisors[i:] are the last k items of the entire list?
        # Yes, because the list is sorted descending globally.
        # So the smallest items in any suffix are the smallest items of the whole list,
        # provided the suffix contains them.
        # The suffix divisors[i:] contains the last k items of the list if i <= num_divs - k.
        # If i > num_divs - k, then the suffix is shorter than the "last k" block?
        # No, if i > num_divs - k, then the number of items in suffix is < k?
        # num_divs - i < k. This is handled by the first check.
        # So if we pass the first check, i <= num_divs - k.
        # Thus the k smallest items in divisors[i:] are exactly divisors[num_divs-k:num_divs].
        return sum(divisors[num_divs-k:num_divs])

    # Wait, is that true?
    # List: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # i=2 (8). k=3.
    # Suffix: [8, 7, 6, 5, 4, 3, 2, 1]
    # Smallest 3: [3, 2, 1]. Sum 6.
    # Formula: sum(divisors[10-3:10]) = sum(divisors[7:10]) = 3+2+1=6. Correct.
    # i=8 (2). k=2.
    # Suffix: [2, 1]
    # Smallest 2: [2, 1]. Sum 3.
    # Formula: sum(divisors[8:10]) = 2+1=3. Correct.
    
    # So min_sum_from(i, k) is constant for all valid i?
    # Yes, as long as i <= num_divs - k.
    # This simplifies things.
    
    min_sum_val = [0] * (num_divs + 1)
    # Precompute sum of last k elements
    # Actually, we can just compute sum(divisors[-k:]) on the fly, it's fast.
    
    def search(idx, rem, k, path):
        if k == 0:
            return rem == 0
        if idx >= num_divs:
            return False
        if rem < 0:
            return False
        
        # Pruning 1: Max possible sum
        # Sum of largest k elements from idx
        if idx + k > num_divs:
            return False
        max_s = sum(divisors[idx:idx+k])
        if max_s < rem:
            return False
            
        # Pruning 2: Min possible sum
        # Sum of smallest k elements in the suffix
        # As derived, this is sum(divisors[num_divs-k:num_divs])
        min_s = sum(divisors[num_divs-k:num_divs])
        if min_s > rem:
            return False

        # Try including divisors[idx]
        # Branch 1: Include
        path.append(divisors[idx])
        if search(idx + 1, rem - divisors[idx], k - 1, path):
            return True
        path.pop()
        
        # Branch 2: Exclude
        # We must ensure we have enough elements left to pick k items
        if num_divs - (idx + 1) >= k:
            if search(idx + 1, rem, k, path):
                return True
        return False

    # Iterate k from 1 to num_divs
    for k in range(1, num_divs + 1):
        path = []
        if search(0, target, k, path):
            return path
            
    return []

# Test
# print(construct())
