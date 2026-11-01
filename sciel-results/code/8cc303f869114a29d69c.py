import time

def construct(n):
    # Hard deadline: must finish well within 150s wall-clock.
    # The recursive construction for n<=7 is extremely fast (<0.1s),
    # so this guard is primarily a safety net.
    deadline = time.time() + 140
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # The standard recursive construction (also known as the "greedy" or "Teodorescu" base)
    # produces a superpermutation of length n! + (n-1)! + ... + 1.
    # For n=4, this is 24+6+2+1 = 33, which is optimal.
    # For n=5, this is 120+24+6+2+1 = 153 (Wait, 5! = 120, 4! = 24, 3!=6, 2!=2, 1!=1. Sum=153? 
    # Actually, the standard construction length is sum_{k=1}^n k!. 
    # Let's re-verify the length formula.
    # S_1 = 1 (len 1)
    # S_2 = 121 (len 3). 1!+2! = 3.
    # S_3 = 123121321 (len 9). 1!+2!+3! = 1+2+6 = 9.
    # S_4 = len 33. 1+2+6+24 = 33.
    # So for n=7, length is 1+2+6+24+120+720+5040 = 5913.
    # The task states the shortest known is 5905. The recursive construction is 5913.
    # The gap is 8.
    
    # The current program implements this recursive construction.
    # The inspirations also implement this.
    # The "inspiration" mentions that for n=4, the hardcoded string "123412314231243121342132413214321" has length 33.
    # Let's check if the recursive construction produces the *same* string or just the same length.
    # For n=4, the recursive construction is deterministic.
    # S_3 = 123121321.
    # Perms of 1,2,3 in S_3:
    # 123, 231, 312, 132, 321, 213.
    # Blocks:
    # 1234123
    # 2314231
    # 3124312
    # 1324132
    # 3214321
    # 2134213
    # Overlaps:
    # 1234123 + 2314231 -> suffix 23, prefix 23. Overlap 2. Append 14231.
    #   Current: 123412314231
    # + 3124312 -> suffix 31, prefix 31. Overlap 2. Append 24312.
    #   Current: 12341231423124312
    # + 1324132 -> suffix 12, prefix 13? No. Suffix 2, prefix 1? No. 
    #   Wait, let's trace carefully.
    #   Current ends in ...24312.
    #   Block is 1324132.
    #   Suffix 2 vs Prefix 1: No.
    #   Suffix 12 vs Prefix 13: No.
    #   Suffix 312 vs Prefix 132: No.
    #   Suffix 4312 vs Prefix 1324: No.
    #   So overlap 0?
    #   Let's check the hardcoded string: 123412314231243121342132413214321
    #   My trace: 12341231423124312 + 1324132 = 123412314231243121324132
    #   Hardcoded: ...243121342132...
    #   It seems my manual trace or the hardcoded string might differ in order or overlap logic?
    #   Actually, the recursive construction is well-defined. The "order of appearance" in S_{m-1} determines the order of permutations.
    #   In S_3 = 123121321:
    #   i=0: 123
    #   i=1: 231
    #   i=2: 312
    #   i=3: 132
    #   i=4: 321
    #   i=5: 213
    #   This matches my list.
    
    # The key improvement to be made is not in the algorithm (which is optimal for n<=4 and near-optimal for n>=5),
    # but in the implementation efficiency and safety.
    # The current code uses string slicing and concatenation, which is O(N^2) in the worst case but N is small.
    # However, for n=7, the string length is ~6000.
    # The loop for m=7 iterates 720 times.
    # Each iteration does an overlap check.
    # The overlap check compares suffix of new_word with prefix of block.
    # Block length is 2*m-1 = 13.
    # So we check up to 13 characters.
    # This is very fast.
    
    # The main risk is if n is larger than 7, but the task says "given as a string of the digits 1..n or a list of integers; the evaluator passes this task's n."
    # And "Shortest superpermutation on seven symbols".
    # So n is likely 7.
    
    # Let's optimize the inner loop slightly to ensure speed.
    # Using a list of integers is often faster than string manipulation in Python for small integers, 
    # but converting to string at the end is required.
    # The current code uses strings.
    
    # Let's stick to the string implementation but ensure it's robust.
    # One potential issue: the overlap check `new_word[-o:] == block[:o]` creates new strings.
    # We can limit the overlap check to `len(block)-1` because if the overlap is `len(block)`, 
    # it means the block is already fully contained? 
    # Actually, if overlap is `len(block)`, then `block[overlap:]` is empty.
    # This happens if the previous block ended with the current block.
    # But since blocks are `P + m + P`, and P is a permutation, they are distinct.
    # However, the suffix of the previous appended part might match the prefix.
    # The max possible overlap is `len(block)`.
    
    # Let's implement the standard recursive construction efficiently.
    
    word = "1"
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, return what we have.
            # For n<=7, this should never happen.
            return word
            
        # Extract unique (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        w_len = len(word)
        m_minus_1 = m - 1
        
        for i in range(w_len - m_minus_1 + 1):
            window = word[i:i + m_minus_1]
            # Check if window is a permutation (all unique)
            if len(set(window)) == m_minus_1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        m_str = str(m)
        
        for p in perms:
            block = p + m_str + p
            block_len = len(block)
            
            # Find max overlap
            # Max possible overlap is min(len(new_word), block_len)
            # But we can optimize: the overlap must be a suffix of new_word and prefix of block.
            # Since block starts with p (a permutation of 1..m-1) and new_word ends with the previous block's suffix,
            # the overlap is likely small.
            
            max_overlap = 0
            limit = min(len(new_word), block_len)
            
            # We check from largest to smallest
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
            
        word = new_word
        
    return word
