import sys

def construct(n):
    # For n <= 4, the standard recursive construction is optimal.
    # For n > 4, it provides the best known constructions (or close to it).
    # The length is sum_{k=1}^n k!
    
    def build_superperm(n):
        if n == 1:
            return [1]
        
        # Get the superpermutation for n-1
        prev = build_superperm(n-1)
        
        # We need to insert n into the superpermutation of 1..n-1
        # The standard construction appends n and then the reverse of the previous sequence?
        # No, the standard construction is:
        # S(n) = S(n-1) + [n] + reverse(S(n-1)) is NOT correct.
        
        # Correct recursive definition:
        # Let S(n-1) be the list of permutations of 1..n-1 in the superperm order.
        # Actually, it's easier to think in terms of the word.
        # S(n) is formed by taking S(n-1), and for each permutation P in S(n-1),
        # we don't just append.
        
        # Let's use the standard algorithm:
        # 1. Start with [1]
        # 2. For m from 2 to n:
        #    - Take the current list of permutations (windows of length m-1)
        #    - For each permutation P of 1..m-1 (in the order they appear in the current superperm):
        #      - Append P, then m, then P again? No.
        
        # The standard construction described in literature:
        # S_1 = 1
        # S_n = S_{n-1} followed by n, followed by S_{n-1} with the last symbol removed? No.
        
        # Let's use the explicit permutation list approach which is easier to code correctly.
        # We maintain a list of permutations.
        # Start with list of perms of length 1: [[1]]
        # Word is [1]
        # For m in 2..n:
        #   new_perms = []
        #   for p in current_perms:
        #       # Insert m into p? No.
        #       # The standard construction inserts the new symbol n at the end of the block for each permutation of n-1?
        #       # Actually, the standard construction is:
        #       # For each permutation p of 1..n-1:
        #       #   Add p to the list?
        #       #   The word extension is: append n, then append p (the same permutation)?
        
        # Let's trace n=3.
        # n=1: Word=[1], Perms=[[1]]
        # n=2:
        #   For p in [[1]]:
        #     Block = [1, 2, 1]?
        #     Word = [1] + [1, 2, 1] with overlap?
        #     Overlap of [1] and [1,2,1] is 1.
        #     New Word = [1, 2, 1].
        #     New Perms? The windows of length 2 in [1,2,1] are (1,2) and (2,1).
        #     So Perms = [[1,2], [2,1]].
        # n=3:
        #   For p in [[1,2], [2,1]]:
        #     Block = [1,2,3,1,2]?
        #     Current Word ends with ...1.
        #     Block starts with 1. Overlap 1.
        #     Append [2,3,1,2].
        #     Word becomes [1,2,1,2,3,1,2].
        #     Next p=[2,1].
        #     Block = [2,1,3,2,1].
        #     Current Word ends with ...2.
        #     Block starts with 2. Overlap 1.
        #     Append [1,3,2,1].
        #     Word becomes [1,2,1,2,3,1,2,1,3,2,1].
        #     Length 11.
        #     Formula: 1! + 2! + 3! = 1 + 2 + 6 = 9?
        #     Wait. 1+2+6=9. My manual trace gave 11.
        #     Let's re-evaluate the standard construction.
        
        # Standard Superpermutation for n=3 is "121321312" (length 9).
        # My trace:
        # n=1: 1
        # n=2: 121 (length 3). Formula 1+2=3. Correct.
        # n=3:
        #   Perms of n=2 in order: (1,2), (2,1).
        #   Word so far: 1 2 1.
        #   For p=(1,2): Block 1 2 3 1 2.
        #     Current end: 1. Block start: 1. Overlap 1.
        #     Append 2 3 1 2.
        #     Word: 1 2 1 2 3 1 2.
        #   For p=(2,1): Block 2 1 3 2 1.
        #     Current end: 2. Block start: 2. Overlap 1.
        #     Append 1 3 2 1.
        #     Word: 1 2 1 2 3 1 2 1 3 2 1.
        #     Length 11.
        #     But optimal is 9.
        #     Why?
        #     The standard construction is NOT simply iterating over the previous permutations and appending p+n+p.
        #     It is:
        #     S(n) = S(n-1) + [n] + S(n-1) with the first n-1 symbols removed? No.
        
        # Let's look at the correct recursive definition.
        # S_1 = 1
        # S_n = S_{n-1} n S_{n-1}^{reverse} ? No.
        
        # Correct construction:
        # Let P_{n-1} be the list of permutations in S_{n-1}.
        # S_n is constructed by:
        # Start with S_{n-1}.
        # For each permutation p in P_{n-1} (in the order they appear in S_{n-1}):
        #   Append n.
        #   Append p.
        #   BUT, we can overlap the appended p with the end of the previous block?
        
        # Let's trace n=3 again with "Append n, then p".
        # S_2 = 1 2 1. Perms: 12, 21.
        # Start S_3 with S_2: 1 2 1.
        # p=12: Append 3, then 12. -> 1 2 1 3 1 2.
        # p=21: Append 3, then 21. -> 1 2 1 3 1 2 3 2 1.
        # Length 9.
        # Does it contain all perms?
        # 121, 213, 131, 312, 123, 232, 321.
        # Perms of 1,2,3:
        # 123 (at index 4? 1,2,3 -> yes, indices 4,5,6? No, 1,2,3 is at 4,5,6?
        # Word: 1 2 1 3 1 2 3 2 1
        # Windows:
        # 121
        # 213
        # 131
        # 312
        # 123 (Yes)
        # 232
        # 321
        # Missing: 132, 213 (have it), 231, 312 (have it), 321 (have it).
        # Wait, 132 is missing?
        # 1 2 1 3 1 2 3 2 1
        # 1,2,1
        # 2,1,3
        # 1,3,1
        # 3,1,2
        # 1,2,3
        # 2,3,2
        # 3,2,1
        # Missing 132, 231, 312? No, 312 is there.
        # 132: 1,3,2. Not there.
        # So "Append n, then p" is NOT the correct standard construction that yields a superpermutation for n>2.
        
        # The correct standard construction (often attributed to Conway or others) is:
        # S(n) = S(n-1) + [n] + (S(n-1) with the first n-1 chars removed)? No.
        
        # Actually, the standard construction that yields length sum k! is:
        # S_1 = [1]
        # S_n:
        #   Take S_{n-1}.
        #   For each permutation p of 1..n-1 (in the order of S_{n-1}):
        #      Append n.
        #      Append p.
        #   This generates a word of length L_{n-1} + n * (n-1) + (n-1)! * (n-1)? No.
        #   Each step adds n + (n-1) symbols? No, overlap.
        
        # Let's check the length of "121312321" (my failed trace).
        # It was 9. But it missed perms.
        
        # Let's look up the actual standard construction.
        # The recursive construction is:
        # S(n) = S(n-1) + n + S(n-1) is false.
        
        # Correctly:
        # Let the permutations of n-1 be P_1, P_2, ..., P_{(n-1)!}.
        # S(n-1) is a word containing all these.
        # S(n) is formed by:
        # Start with S(n-1).
        # For each P_i:
        #   Append n.
        #   Append P_i.
        #   However, the end of the previous P_{i-1} + n + P_{i-1} block might overlap with n + P_i?
        #   The end of the previous block is P_{i-1}.
        #   The new block starts with n.
        #   So no overlap between P_{i-1} and n.
        #   So we just append n and P_i.
        #   Total added length per permutation: 1 + (n-1) = n.
        #   Total length = L_{n-1} + (n-1)! * n.
        #   L_n = L_{n-1} + n * (n-1)! = L_{n-1} + n!.
        #   L_1 = 1.
        #   L_n = sum_{k=1}^n k!.
        
        # Does this work?
        # Let's re-trace n=3 with this logic.
        # S_2 = 1 2 1. Perms: 12, 21.
        # S_3:
        # Start 1 2 1.
        # P_1 = 12. Append 3, 12. -> 1 2 1 3 1 2.
        # P_2 = 21. Append 3, 21. -> 1 2 1 3 1 2 3 2 1.
        # Length 9.
        # I previously said it missed 132. Let's check again.
        # Word: 1 2 1 3 1 2 3 2 1
        # Windows:
        # 121
        # 213
        # 131
        # 312
        # 123
        # 232
        # 321
        # It seems I only listed 7 windows for a length 9 word?
        # Length 9 has 9-3+1 = 7 windows.
        # There are 6 permutations. So 7 windows is enough to cover 6 perms if one is duplicate or if I missed one.
        # Wait, 7 windows for 6 perms means one perm must be repeated or I missed a window.
        # Windows:
        # 0-2: 121
        # 1-3: 213
        # 2-4: 131
        # 3-5: 312
        # 4-6: 123
        # 5-7: 232
        # 6-8: 321
        # Perms present: 121(no), 213, 131(no), 312, 123, 232(no), 321.
        # Valid perms: 213, 312, 123, 321.
        # Missing: 132, 231.
        # So this construction DOES NOT work for n=3.
        
        # Why? Because the "standard" construction I am thinking of is for the "factoradic" ordering which works differently.
        # The construction that yields sum k! is:
        # S(n) is built from S(n-1) by inserting n into the "gaps" in a specific way?
        
        # Actually, the known optimal construction for n=3 is 121321312 (length 9).
        # Let's see how that is built.
        # 121
        # 321? No.
        
        # Let's use a different, robust method:
        # For n <= 7, we can use the pre-computed standard superpermutations or generate them via the correct recursive definition.
        # The correct recursive definition for the "standard" superpermutation (length sum k!) is:
        # S_1 = [1]
        # S_n = S_{n-1}
        # For each permutation p of 1..n-1 (in the order they appear in S_{n-1}):
        #    Append n
        #    Append p
        #    BUT, we must ensure that the sequence of permutations is correct.
        #    The issue above was that the resulting word didn't contain all perms.
        #    Why? Because the "order" of permutations in S_{n-1} matters.
        #    In S_2 = 121, the perms are 12, 21.
        #    In S_3, if we use 12, 21:
        #    121 + 312 + 321 = 121312321.
        #    This missed 132, 231.
        
        # What if the order was different?
        # If S_2 was 212? Perms 21, 12.
        # 212 + 321 + 312 = 212321312.
        # Windows: 212, 123, 232, 321, 213, 131, 312.
        # Valid: 123, 321, 213, 312.
        # Missing 132, 231.
        
        # It seems the simple "append n, append p" does NOT produce a superpermutation for n=3.
        # The length formula sum k! is correct for the *length* of the standard superpermutation, but the construction is more subtle.
        
        # The correct construction:
        # S(n) = S(n-1) + n + S(n-1) is wrong.
        # It is:
        # S(n) = S(n-1)
        # For each p in Perms(n-1):
        #    Append n
        #    Append p
        #    ... wait, I just did that.
        
        # Let's check a known source.
        # "The standard superpermutation for n=3 is 121321312."
        # Let's decompose 121321312.
        # 121
        # 321?
        # 312?
        
        # Actually, the construction is:
        # S(n) = S(n-1) + n + S(n-1) with the last n-1 symbols of S(n-1) removed? No.
        
        # Let's try to generate it via the "permutation list" method but correctly.
        # The list of permutations for n=3 in the standard superperm 121321312 is:
        # 121 (no)
        # 213
        # 132
        # 321
        # 213 (no)
        # 132 (no)
        # 312
        # Wait, 121321312:
        # 121
        # 213
        # 132
        # 321
        # 213
        # 132
        # 312
        # Perms: 213, 132, 321, 312. Missing 123, 231.
        # So 121321312 is NOT a superpermutation?
        # Let's check 123.
        # 1 2 1 3 2 1 3 1 2
        # 121
        # 213
        # 132
        # 321
        # 213
        # 132
        # 312
        # Yes, 123 is missing.
        
        # Okay, I am confusing myself. The standard superpermutation for n=3 IS length 9.
        # One such is 121321312? No, that one misses 123.
        # Let's find a valid one.
        # 123121321?
        # 123, 231, 312, 121, 213, 132, 321.
        # 123 (Yes)
        # 231 (Yes)
        # 312 (Yes)
        # 121 (No)
        # 213 (Yes)
        # 132 (Yes)
        # 321 (Yes)
        # This works! Length 9.
        
        # How to generate 123121321?
        # 123
        # 121?
        
        # Let's stick to the simplest robust method:
        # For n <= 7, we can use a pre-computed list or a very fast recursive generator that is known to be correct.
        # Given the risk of bugs in hand-rolled recursion, and the fact that the current code *does* generate a valid standard superperm (it just optimizes it poorly), I will replace the optimization part with a fast, correct standard generator and remove the slow hill-climbing.
        
        # The current code's `get_standard_superperm` actually works correctly for generating a valid superperm (it uses the "block" method which is correct).
        # Let's verify `get_standard_superperm` logic from the current code:
        # It iterates m from 2 to n.
        # It extracts windows of length m-1 from the current word.
        # It creates blocks `p + [m] + p`.
        # It appends them with maximum overlap.
        # This is the correct standard construction.
        
        # So, I will simply return the result of `get_standard_superperm(n)` without the slow optimization.
        # This will produce a valid word of length sum k!.
        # For n=7, length 5913.
        # This is a valid, fast, deterministic solution.
        
        pass

    # Implementation
    def get_standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                # Calculate max overlap between new and block
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    word = get_standard_superperm(n)
    return "".join(map(str, word))
