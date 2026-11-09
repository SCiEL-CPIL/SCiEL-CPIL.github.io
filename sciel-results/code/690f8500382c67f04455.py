import time
import random

N = 16

def _verify_network(comps):
    """
    Verifies if the network sorts all 2^N binary inputs.
    Uses bitmask simulation for speed.
    """
    for mask in range(1 << N):
        state = mask
        for i, j in comps:
            # Check if i has 1 and j has 0
            # i < j
            if (state >> i) & 1 and not ((state >> j) & 1):
                # Swap bits i and j
                state ^= (1 << i) | (1 << j)
        
        # Check if sorted: all 0s then 1s
        # Count 1s
        k = bin(state).count('1')
        if k > 0:
            # The lowest N-k bits must be 0
            if state & ((1 << (N - k)) - 1):
                return False
    return True

def _batcher_odd_even_merge_sort(n):
    """
    Generate Batcher's Odd-Even Merge Sort network for n inputs.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    comps = []
    # Recursively sort first half and second half
    # First half: wires 0 to n/2 - 1
    # Second half: wires n/2 to n - 1
    half = n // 2
    comps.extend(_batcher_odd_even_merge_sort(half))
    for i, j in _batcher_odd_even_merge_sort(half):
        comps.append((i + half, j + half))
    
    # Merge the two sorted halves
    comps.extend(_odd_even_merge(n))
    
    return comps

def _odd_even_merge(n):
    """
    Generate the odd-even merge step for n elements (assumes two sorted halves of n/2).
    """
    if n == 1:
        return []
    
    comps = []
    half = n // 2
    
    # Odd-even merge:
    # 1. Compare elements i and i + n/2 for i in range(n/2)
    #    But we need to interleave properly.
    # Standard Odd-Even Merge:
    # Step 1: Compare (i, i + n/2) for i = 0..n/2-1
    for i in range(half):
        comps.append((i, i + half))
    
    # Step 2: Recursively merge odd and even positions
    # The elements at positions 0, 2, 4... form one subsequence
    # The elements at positions 1, 3, 5... form another
    # We need to map these back to wire indices.
    # After step 1, wire i holds min(original[i], original[i+n/2])
    # wire i+n/2 holds max(original[i], original[i+n/2])
    
    # The even positions in the final array correspond to:
    # Original indices: 0, 2, 4... from the combined array?
    # Actually, the odd-even merge algorithm works on the indices 0..n-1.
    # We compare (i, i+n/2).
    # Then we recursively sort the even indices (0, 2, 4...) and odd indices (1, 3, 5...) of the current array.
    # But the wires are still 0..n-1.
    # The values that need to be sorted are now distributed such that:
    # Even positions in the logical array contain values that came from the "min" outputs of the first half?
    # No, that's not right.
    
    # Let's look at the standard definition.
    # Odd-Even Merge Sort:
    # 1. Split into two halves.
    # 2. Recursively sort each half.
    # 3. Merge the two sorted halves using Odd-Even Merge.
    
    # Odd-Even Merge of two sorted lists A[0..n/2-1] and B[0..n/2-1]:
    # 1. Compare A[i] and B[i] for all i. Let C[i] = min(A[i], B[i]), D[i] = max(A[i], B[i]).
    #    (This is what we did above with (i, i+n/2) if we map A to 0..n/2-1 and B to n/2..n-1)
    # 2. Recursively odd-even merge the sequence C[0], C[1], ... C[n/2-1] and D[0], D[1], ... D[n/2-1]?
    #    No, that's not right either.
    
    # Correct Odd-Even Merge:
    # Input: Two sorted sequences of length n/2.
    # 1. Compare element i of first seq with element i of second seq.
    #    Let low[i] = min, high[i] = max.
    # 2. The resulting sequence is formed by interleaving? No.
    #    The sequence of lows is sorted? No.
    #    The sequence of highs is sorted? No.
    #    
    #    Actually, the standard algorithm is:
    #    1. Compare (i, i + n/2) for i in 0..n/2-1.
    #    2. Recursively apply odd-even merge to the even-indexed elements (0, 2, 4...) of the result?
    #       And odd-indexed elements (1, 3, 5...)?
    #       
    #    Wait, the elements at even indices (0, 2, ...) in the *logical* array after step 1 are not simply the lows.
    #    Let's trace.
    #    A = [a0, a1, a2, a3], B = [b0, b1, b2, b3].
    #    Wires: 0:a0, 1:a1, 2:a2, 3:a3, 4:b0, 5:b1, 6:b2, 7:b3.
    #    Step 1: Compare (0,4), (1,5), (2,6), (3,7).
    #    Wire 0: min(a0, b0), Wire 4: max(a0, b0)
    #    Wire 1: min(a1, b1), Wire 5: max(a1, b1)
    #    Wire 2: min(a2, b2), Wire 6: max(a2, b2)
    #    Wire 3: min(a3, b3), Wire 7: max(a3, b3)
    #    
    #    Now we want to sort the 8 elements.
    #    The odd-even merge says:
    #    Recursively merge the sequence of elements at even positions in the *original* indices?
    #    No, it says:
    #    "Recursively merge the even-indexed elements and the odd-indexed elements of the combined sequence."
    #    But which elements are at even/odd positions?
    #    The combined sequence is the 8 wires.
    #    Even wires: 0, 2, 4, 6.
    #    Odd wires: 1, 3, 5, 7.
    #    
    #    Are the values in even wires sorted relative to each other?
    #    Wire 0 < Wire 2 < Wire 4 < Wire 6?
    #    min(a0,b0) < min(a2,b2)? Yes, because a0<a2 and b0<b2.
    #    So the subsequence of even wires is sorted? No, we need to merge them further?
    #    Wait, if they are already sorted, why merge?
    #    
    #    Let's check the literature.
    #    "Odd-even merge sort"
    #    Step 1: Compare (i, i + n/2).
    #    Step 2: Recursively merge the even-indexed elements (0, 2, 4...) and odd-indexed elements (1, 3, 5...) of the *result*.
    #    
    #    But the result of step 1 is NOT two sorted sequences.
    #    The even-indexed elements are sorted?
    #    Let's check: min(a0,b0), min(a2,b2), max(a0,b0), max(a2,b2).
    #    Is min(a0,b0) < min(a2,b2)? Yes.
    #    Is min(a2,b2) < max(a0,b0)? Not necessarily.
    #    Example: a=[1, 10], b=[5, 6].
    #    min(1,5)=1, min(10,6)=6, max(1,5)=5, max(10,6)=10.
    #    Even wires: 1, 6. Odd wires: 5, 10.
    #    Sorted even: 1, 6. Sorted odd: 5, 10.
    #    Final sorted: 1, 5, 6, 10.
    #    If we just take even then odd: 1, 6, 5, 10. Not sorted.
    #    So we need to merge the even and odd subsequences?
    #    No, the recursive step is:
    #    "Apply odd-even merge to the even-indexed elements" -> This implies they are NOT sorted yet?
    #    Or does it mean we treat the even-indexed elements as two sorted halves?
    #    
    #    Ah, the trick is:
    #    After step 1, the even-indexed elements form a sorted sequence? No.
    #    The elements at positions 0, 2, 4, 6 are: min(a0,b0), min(a2,b2), max(a0,b0), max(a2,b2).
    #    This is NOT sorted.
    #    
    #    Let's re-read the algorithm carefully.
    #    "The odd-even merge of two sorted lists of length n/2 is defined as follows:
    #     1. Compare element i of the first list with element i of the second list.
    #     2. Recursively merge the even-indexed elements and the odd-indexed elements of the result."
    #     
    #    Wait, "even-indexed elements of the result" refers to indices 0, 2, 4... in the *combined* array of length n.
    #    But are they sorted?
    #    
    #    Actually, there is a property:
    #    The even-indexed elements of the result are sorted?
    #    Let's check:
    #    Result[0] = min(a0, b0)
    #    Result[2] = min(a2, b2)
    #    Result[4] = max(a0, b0)
    #    Result[6] = max(a2, b2)
    #    
    #    Is Result[0] <= Result[2]? Yes.
    #    Is Result[2] <= Result[4]?
    #    min(a2, b2) <= max(a0, b0)?
    #    Since a2 >= a0 and b2 >= b0, min(a2, b2) >= min(a0, b0).
    #    But is min(a2, b2) <= max(a0, b0)?
    #    If a0=1, b0=100, a2=50, b2=60.
    #    min(50, 60) = 50. max(1, 100) = 100. 50 <= 100. Yes.
    #    If a0=50, b0=60, a2=1, b2=100. (But a and b must be sorted, so a2>=a0, b2>=b0).
    #    So a2>=50, b2>=60.
    #    min(a2, b2) >= 50. max(a0, b0) = 60.
    #    Is min(a2, b2) <= 60?
    #    If a2=51, b2=100, min=51 <= 60.
    #    If a2=70, b2=80, min=70 > 60.
    #    So Result[2] can be > Result[4].
    #    So the even-indexed elements are NOT sorted.
    #    
    #    Then what does "recursively merge the even-indexed elements" mean?
    #    It means we take the sequence of even-indexed elements, which is composed of two sorted sequences?
    #    Sequence: min(a0,b0), min(a2,b2), max(a0,b0), max(a2,b2).
    #    This is not two sorted sequences.
    #    
    #    I think I am misinterpreting the indices.
    #    The standard Odd-Even Merge works on the *values*, not just the wires.
    #    The recursive call is on the subsequences of *values* at even and odd positions.
    #    But we are building a network on fixed wires.
    #    
    #    Let's look at a concrete implementation of Batcher's Odd-Even Merge Sort network generation.
    #    
    #    def odd_even_merge(n):
    #        if n == 1: return []
    #        comps = []
    #        # Step 1: Compare i and i + n/2
    #        for i in range(n // 2):
    #            comps.append((i, i + n // 2))
    #        
    #        # Step 2: Recursively merge even and odd positions
    #        # The even positions are 0, 2, 4, ...
    #        # The odd positions are 1, 3, 5, ...
    #        # We need to map these to a smaller problem.
    #        # The elements at even positions are: wire 0, wire 2, wire 4, ...
    #        # We need to sort these n/2 elements.
    #        # But they are not sorted. They are the result of comparing two sorted lists.
    #        # Actually, the property is that the even-indexed elements are sorted?
    #        # No, I showed they are not.
    #        
    #        # Wait, I found a reference:
    #        # "The odd-even merge of two sorted lists A and B of length n/2 is:
    #        # 1. Compare A[i] and B[i] for i=0..n/2-1.
    #        # 2. Recursively merge the even-indexed elements of the result and the odd-indexed elements of the result."
    #        
    #        # This implies that the even-indexed elements of the result form a sequence that can be merged?
    #        # Or that they are already sorted?
    #        # If they were already sorted, we wouldn't need to merge.
    #        # "Merge" implies they are two sorted lists.
    #        
    #        # Ah! The even-indexed elements of the result are:
    #        # R[0], R[2], R[4], ..., R[n-2]
    #        # These are NOT two sorted lists.
    #        
    #        # Let's look at the structure again.
    #        # Maybe the recursion is on the *indices* 0..n/2-1?
    #        # No.
    #        
    #        # Let's try a different approach.
    #        # I will use a known correct implementation or derive it carefully.
    #        
    #        # From "Sorting Networks" by Knuth:
    #        # Odd-Even Merge Sort:
    #        # 1. Split into two halves.
    #        # 2. Recursively sort each half.
    #        # 3. Merge the two sorted halves.
    #        
    #        # Merge(A, B) where A, B are sorted:
    #        # 1. Compare A[i] and B[i]. Let C[i] = min, D[i] = max.
    #        # 2. Recursively merge C[0], C[2], C[4]... and C[1], C[3], C[5]...?
    #        #    No, that's not right.
    #        
    #        # Correct Merge:
    #        # 1. Compare A[i] and B[i].
    #        # 2. The result is a sequence of 2n elements.
    #        # 3. Recursively merge the even-indexed elements of the result and the odd-indexed elements of the result.
    #        
    #        # This is confusing. Let's look at a Python implementation online.
    #        
    #        # Found this logic:
    #        # def merge(n):
    #        #   if n == 1: return []
    #        #   comps = []
    #        #   for i in range(n//2):
    #        #      comps.append((i, i+n//2))
    #        #   # Recursively merge even and odd
    #        #   # Even indices: 0, 2, 4...
    #        #   # Odd indices: 1, 3, 5...
    #        #   # We need to call merge on n/2 elements.
    #        #   # But which elements?
    #        #   # The even-indexed elements are at wires 0, 2, 4...
    #        #   # We need to sort them.
    #        #   # But they are not sorted.
    #        #   
    #        #   # Wait, I think the key is that the even-indexed elements are sorted?
    #        #   # Let's re-verify.
    #        #   # A = [1, 3], B = [2, 4].
    #        #   # Compare (0,2): 1,2 -> 1,2. (Wire 0:1, Wire 2:2)
    #        #   # Compare (1,3): 3,4 -> 3,4. (Wire 1:3, Wire 3:4)
    #        #   # Wires: 0:1, 1:3, 2:2, 3:4.
    #        #   # Even wires: 0:1, 2:2. Sorted.
    #        #   # Odd wires: 1:3, 3:4. Sorted.
    #        #   
    #        #   # A = [1, 10], B = [5, 6].
    #        #   # Compare (0,2): 1,5 -> 1,5. (Wire 0:1, Wire 2:5)
    #        #   # Compare (1,3): 10,6 -> 6,10. (Wire 1:6, Wire 3:10)
    #        #   # Wires: 0:1, 1:6, 2:5, 3:10.
    #        #   # Even wires: 0:1, 2:5. Sorted.
    #        #   # Odd wires: 1:6, 3:10. Sorted.
    #        #   
    #        #   # It seems the even-indexed elements ARE sorted!
    #        #   # Let's prove it.
    #        #   # R[2k] = min(A[k], B[k]).
    #        #   # R[2k+1] = max(A[k], B[k]).
    #        #   # We want to show R[2k] <= R[2k+2].
    #        #   # min(A[k], B[k]) <= min(A[k+1], B[k+1]).
    #        #   # Since A[k] <= A[k+1] and B[k] <= B[k+1], this is true.
    #        #   # So the even-indexed elements are sorted!
    #        #   # Similarly, the odd-indexed elements are sorted!
    #        #   
    #        #   # So, after step 1, we have two sorted sequences:
    #        #   # Even wires: 0, 2, 4, ... (sorted)
    #        #   # Odd wires: 1, 3, 5, ... (sorted)
    #        #   
    #        #   # But we want the final sequence to be sorted: 0, 1, 2, 3, ...
    #        #   # We need to merge these two sorted sequences (even wires and odd wires) into one sorted sequence.
    #        #   # But they are interleaved in the wires.
    #        #   # Wire 0 (even), Wire 1 (odd), Wire 2 (even), Wire 3 (odd)...
    #        #   # We need to sort the values in wires 0..n-1.
    #        #   # The values in even wires are sorted. The values in odd wires are sorted.
    #        #   # We need to merge them.
    #        #   # This is exactly the same problem as merging two sorted lists!
    #        #   # But the lists are not contiguous in the wires.
    #        #   # We can't just call merge(n/2) directly because merge expects contiguous wires.
    #        #   
    #        #   # However, we can map the even wires to a new index space 0..n/2-1 and odd wires to 0..n/2-1?
    #        #   # No, we need to perform comparators on the actual wires.
    #        #   
    #        #   # The recursive step is:
    #        #   # "Recursively merge the even-indexed elements and the odd-indexed elements".
    #        #   # This means we apply the merge network to the even-indexed elements (treating them as a contiguous list) and to the odd-indexed elements (treating them as a contiguous list).
    #        #   # But how do we implement this on the fixed wires?
    #        #   # We need to generate comparators that sort the even wires and the odd wires.
    #        #   # Since the even wires are already sorted, do we need to do anything?
    #        #   # No, the recursive merge is to merge the two sorted sequences (even and odd) into one sorted sequence.
    #        #   # Wait, if even and odd are already sorted, and we want to merge them, we need to compare elements from the even sequence with elements from the odd sequence.
    #        #   
    #        #   # Actually, the standard algorithm says:
    #        #   # "Recursively merge the even-indexed elements and the odd-indexed elements of the result."
    #        #   # This implies that the even-indexed elements are NOT sorted?
    #        #   # Or that we need to merge them with each other?
    #        #   # No, "merge the even-indexed elements" is ambiguous.
    #        #   # It likely means "apply the merge network to the subsequence of even-indexed elements".
    #        #   # But if they are already sorted, the merge network would do nothing?
    #        #   # No, the merge network assumes two sorted halves.
    #        #   
    #        #   # I think the correct interpretation is:
    #        #   # The result of step 1 is two sorted sequences: E (even wires) and O (odd wires).
    #        #   # We want to merge E and O into a single sorted sequence.
    #        #   # This is done by recursively applying the merge procedure to E and O.
    #        #   # But E and O are interleaved.
    #        #   
    #        #   # Let's look at the comparators generated by a known correct implementation.
    #        #   
    #        #   # I will use a simpler approach:
    #        #   # I will use Batcher's Bitonic Sort, which is easier to implement correctly.
    #        #   # Bitonic sort uses 4*log2(n) depth and O(n log^2 n) comparators.
    #        #   # For n=16, it uses 4*4*16/2 = 128? No, n/2 * log^2 n = 8 * 16 = 128.
    #        #   # Odd-even transposition uses 120.
    #        #   # So Bitonic is worse.
    #        #   
    #        #   # Let's stick to Odd-Even Merge Sort.
    #        #   # I found a correct implementation online:
    #        #   
    #        #   def odd_even_merge(n):
    #        #       if n == 1:
    #        #           return []
    #        #       comps = []
    #        #       for i in range(n // 2):
    #        #           comps.append((i, i + n // 2))
    #        #       
    #        #       # Recursively merge even and odd
    #        #       # Even indices: 0, 2, 4...
    #        #       # Odd indices: 1, 3, 5...
    #        #       # We need to generate comparators for sorting the even-indexed elements and odd-indexed elements.
    #        #       # But they are already sorted!
    #        #       # So why recurse?
    #        #       
    #        #       # Ah, I see. The recursive step is to merge the even-indexed elements *with each other*? No.
    #        #       # It is to merge the even-indexed elements *and* the odd-indexed elements *into a single sorted sequence*.
    #        #       # But they are already sorted separately.
    #        #       
    #        #       # I think the key is that the "merge" operation is defined recursively on the *indices*.
    #        #       # The even-indexed elements are at positions 0, 2, 4...
    #        #       # We can map them to 0, 1, 2...
    #        #       # And the odd-indexed elements to 0, 1, 2...
    #        #       # Then we apply the merge network to these mapped indices?
    #        #       # But the merge network compares (i, i + n/2).
    #        #       # If we map even indices to 0..n/2-1, then comparing (i, i+n/2) in the mapped space corresponds to comparing (2i, 2i+n) in the original space?
    #        #       # No, that's not right.
    #        #       
    #        #       # Let's try a different tactic.
    #        #       # I will use the fact that the known optimal network for 16 inputs is 60 comparators.
    #        #       # I will start with a known good network (e.g., from a database) and then prune it.
    #        #       # But I don't have a database.
    #        #       
    #        #       # I will use a random search to find a good network.
    #        #       # I will start with a random network and try to improve it.
    #        #       # But random networks are unlikely to be correct.
    #        #       
    #        #       # I will use a genetic algorithm.
    #        #       # But that might be too slow.
    #        #       
    #        #       # I will use the fact that the current program uses random deletion from a 120-comparator network.
    #        #       # I will improve the verification speed and the deletion strategy.
    #        #       
    #        #       # To speed up verification, I can use NumPy.
    #        #       # I can represent the state as a NumPy array and perform vectorized operations.
    #        #       
    #        #       # Let's try that.
    #        #       
    #        #   return comps
    #        
    #    return comps

def construct():
    # Use a simpler, known-correct network as a starting point.
    # I will use a simple bubble sort network (which is the same as odd-even transposition).
    # Then I will try to prune it using a smarter strategy.
    
    # Start with Odd-Even Transposition Network (120 comparators)
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Use a smarter pruning strategy:
    # 1. Try to remove comparators in reverse order (from the end).
    # 2. If a comparator cannot be removed, try to move it to a later position?
    # 3. Use a random restart if stuck.
    
    # For now, let's just improve the verification speed.
    # I will use a bitmask approach, but optimize it.
    
    # The current verification is O(2^N * C).
    # For N=16, 2^16 = 65536.
    # C is at most 120.
    # 65536 * 120 = 7.8 million operations.
    # This should be fast enough in Python?
    # Let's test.
    
    # Actually, the current verification is already using bitmask.
    # The bottleneck is the inner loop.
    
    # Let's try to remove comparators in a specific order.
    # I will try to remove comparators that are "redundant" based on a heuristic.
    # Heuristic: If a comparator (i, j) is such that i and j are "close" and there are other comparators between them, it might be redundant.
    
    # For now, I will stick to random deletion but with a better verification.
    
    while time.time() < deadline:
        if not comps:
            break
            
        # Pick a random comparator to remove
        idx = random.randint(0, len(comps) - 1)
        candidate = comps[idx]
        
        # Remove it
        comps.pop(idx)
        
        # Verify
        if _verify_network(comps):
            # Keep it removed
            pass
        else:
            # Restore it
            comps.insert(idx, candidate)
            
    return comps
