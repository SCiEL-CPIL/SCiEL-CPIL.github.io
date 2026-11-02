import time
import math
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    timeout = 145.0  # 5s margin before 150s limit
    
    # 1. Generate standard superpermutation using recursive construction
    # S_1 = [1]
    # S_k = (S_{k-1} with k inserted in appropriate places)
    # Standard formula: S_k is formed by taking the list of permutations of 1..k-1
    # in the order they appear in S_{k-1}, and for each permutation P,
    # appending P, then k, then P again? No, that's not quite right.
    # 
    # Correct recursive construction:
    # Let P_1, ..., P_m be the permutations of 1..k-1 in the order they appear in S_{k-1}.
    # S_k = P_1, k, P_2, k, ..., P_m, k
    # But we can overlap the end of P_i with the start of P_{i+1}?
    # Actually, the standard construction S_k has length k * (k-1)! - (k-2)! ... no.
    # Length is sum_{i=1}^k i! ? No.
    # For n=1, L=1.
    # For n=2, S_2 = 1, 2, 1. L=3. (1! + 2! = 3? No, 1+2=3. Wait. 1! = 1, 2! = 2. Sum = 3.)
    # For n=3, S_3 = 1, 2, 3, 1, 2, 1, 3, 2, 1. L=9.
    # Sum 1! + 2! + 3! = 1 + 2 + 6 = 9.
    # For n=4, L = 1 + 2 + 6 + 24 = 33.
    # General formula: L_n = sum_{i=1}^n i!.
    
    def build_superperm(k):
        if k == 1:
            return [1]
        prev = build_superperm(k - 1)
        # Extract permutations of 1..k-1 from prev
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 1) + 1):
            window = tuple(prev[i:i + k - 1])
            if window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            new_word.extend(p)
            new_word.append(k)
            # Note: The next permutation will start with the same prefix?
            # No, the standard construction appends the next permutation.
            # Overlap: The last k-1 elements of p are NOT necessarily the first k-1 elements of the next p.
            # However, in the standard construction, the list of perms is such that 
            # consecutive perms share a long suffix/prefix?
            # Actually, the simple concatenation P_1, k, P_2, k, ... works if we don't overlap P_i and P_{i+1}.
            # But wait, if we just concatenate, do we get a valid superperm?
            # Yes, because every permutation of 1..k is of the form P, k or k, P? No.
            # Every permutation of 1..k contains k. Remove k, we get a permutation of 1..k-1.
            # In S_k, we have ... P_i, k, P_{i+1} ...
            # The windows of length k are:
            # 1. Windows entirely within P_i? No, P_i is length k-1.
            # 2. Windows crossing the boundary.
            # Let's trust the sum of factorials formula.
            # The construction S_k = S_{k-1} k S_{k-1} is NOT correct for length.
            # The correct one is:
            # S_k = (S_{k-1})_1, k, (S_{k-1})_2, k, ...
            # Where (S_{k-1})_i is the i-th permutation of 1..k-1?
            
            # Let's just use the known property that the standard construction yields length sum(i!).
            # To implement it correctly:
            # S_k is built by taking the list of permutations of 1..k-1 (in order of appearance in S_{k-1})
            # and inserting k after each one.
            # But we must handle overlaps if any.
            # In the standard construction, there are NO overlaps between the permutation blocks P_i and P_{i+1}
            # other than the fact that they are just listed sequentially?
            # Let's check n=3.
            # S_2 = [1, 2, 1]. Perms: (1,2), (2,1).
            # P_1 = (1,2). Append 3. -> [1, 2, 3]
            # P_2 = (2,1). Append 3. -> [2, 1, 3]
            # Concatenate: [1, 2, 3, 2, 1, 3]
            # Length 6. But S_3 is length 9.
            # Where are the missing windows?
            # S_3 should contain 123, 231, 312, 132, 213, 321.
            # [1, 2, 3, 2, 1, 3] windows:
            # 123, 232 (no), 321, 213.
            # Missing: 231, 312, 132.
            # So simple concatenation of P_i, k is NOT sufficient.
            
            # The correct recursive step is:
            # S_k = S_{k-1}, k, S_{k-1}? No, that doubles length.
            
            # Let's look at S_3 = 1 2 3 1 2 1 3 2 1.
            # It looks like S_2, 3, S_2?
            # S_2 = 1 2 1.
            # 1 2 1, 3, 1 2 1 -> 1 2 1 3 1 2 1. Length 7. Still not 9.
            
            # Actually, the standard construction is:
            # List all permutations of 1..k-1.
            # For each permutation P, output P, then k.
            # But to ensure ALL permutations of 1..k are covered, the list of P's must be such that
            # consecutive P's overlap significantly?
            
            # Let's use a simpler, brute-force-ish construction for small n that is known to work.
            # Or just use the fact that we can generate a valid superperm by:
            # 1. List all permutations of 1..n.
            # 2. Find a Hamiltonian path in the de Bruijn graph? No, that's for binary/cyclic.
            
            # Let's use the recursive definition properly.
            # S_n is a superpermutation of 1..n.
            # S_n = S_{n-1} n S_{n-1} is a superpermutation but not shortest.
            # Length is 2 * L_{n-1} + 1.
            # L_1 = 1.
            # L_2 = 2*1 + 1 = 3. (1 2 1) Correct.
            # L_3 = 2*3 + 1 = 7. (1 2 1 3 1 2 1).
            # Windows: 121, 213, 131, 312, 121.
            # Perms: 121 (no), 213, 131 (no), 312, 121 (no).
            # This is NOT a valid superpermutation of length 3 because it doesn't contain 123.
            # So S_n = S_{n-1} n S_{n-1} is WRONG.
            
            # Correct standard construction:
            # S_1 = 1
            # S_k = S_{k-1} k S_{k-1} ... wait.
            
            # Let's hardcode the generation for n=7 using the known algorithm:
            # The sequence of permutations of 1..k-1 in S_{k-1} is P_1, ..., P_m.
            # S_k is formed by:
            # P_1, k, P_2, k, ..., P_m, k
            # BUT, we can overlap the end of P_i with the start of P_{i+1}?
            # In the standard construction, P_i and P_{i+1} share a suffix of length k-2?
            # No, they are just distinct permutations.
            
            # Actually, the length of the standard superpermutation is sum_{i=1}^n i!.
            # This is achieved by the following recursive step:
            # S_n = S_{n-1}
            # For each permutation P of 1..n-1 in the order they appear in S_{n-1}:
            #   Append n
            #   Append P
            #   (With overlap if possible?)
            
            # Let's try this:
            # S_1 = [1]
            # S_2:
            #   Perms of S_1: [1]
            #   Append 2, Append 1.
            #   S_2 = [1, 2, 1]. Correct.
            # S_3:
            #   Perms of S_2: [1,2], [2,1]
            #   Start with S_2: [1, 2, 1]
            #   For P=[1,2]: Append 3, Append 1,2.
            #   Word: [1, 2, 1, 3, 1, 2]
            #   For P=[2,1]: Append 3, Append 2,1.
            #   Word: [1, 2, 1, 3, 1, 2, 3, 2, 1]
            #   Length 9.
            #   Windows: 123, 231, 312, 121(no), 213, 132(no), 321.
            #   Perms: 123, 231, 312, 213, 321. Missing 132.
            #   So this construction is also flawed if we don't handle overlaps or ordering correctly.
            
            # The correct ordering of permutations in S_{n-1} is crucial.
            # The standard construction ensures that consecutive permutations in the list
            # are such that the last n-2 elements of P_i are the first n-2 elements of P_{i+1}.
            # This allows an overlap of n-2 characters.
            
            # Since implementing the correct permutation ordering is complex,
            # and we only need a VALID superpermutation to start with (not necessarily the shortest standard one,
            # though the standard one is a good target),
            # we can use a simpler, slightly longer construction if needed,
            # OR we can just use the sum of factorials formula and trust that
            # a known algorithm exists.
            
            # However, for the purpose of this coding task, I will use a robust method:
            # Generate a valid superpermutation by simply concatenating all permutations
            # with overlaps.
            # This might be longer, but it's easy to implement correctly.
            # Then we shorten it.
            
            # Even simpler:
            # Just return a string of all permutations concatenated with 1 overlap?
            # No, that might not be valid.
            
            # Let's use the recursive construction that IS correct:
            # S_1 = "1"
            # S_k = S_{k-1} + k + S_{k-1} is WRONG.
            
            # The correct one is:
            # S_k = S_{k-1}
            # For each permutation P of 1..k-1 (in the specific order generated by S_{k-1}):
            #   Append k
            #   Append P
            #   (Overlap the end of current S_k with start of P? No, P is appended after k.)
            
            # Let's look at S_3 again: 1 2 3 1 2 1 3 2 1.
            # S_2 = 1 2 1.
            # Perms in S_2: 12, 21.
            # Step 1: Start with S_2: 1 2 1.
            # Step 2: For P=12: Append 3, Append 12.
            #          1 2 1 3 1 2.
            # Step 3: For P=21: Append 3, Append 21.
            #          1 2 1 3 1 2 3 2 1.
            # This matches S_3!
            # Why did I think 132 was missing?
            # Windows of 1 2 1 3 1 2 3 2 1:
            # 121 (no)
            # 213 (yes)
            # 131 (no)
            # 312 (yes)
            # 123 (yes) -- Wait, 123 is at index 0? 1,2,1. No.
            # Let's list windows of 1 2 1 3 1 2 3 2 1:
            # i=0: 121
            # i=1: 213
            # i=2: 131
            # i=3: 312
            # i=4: 123
            # i=5: 232
            # i=6: 321
            # Perms: 213, 312, 123, 321. Missing 132, 231.
            
            # So S_3 generated this way is NOT a superpermutation.
            # The standard S_3 is 1 2 3 1 2 1 3 2 1.
            # My generated one: 1 2 1 3 1 2 3 2 1.
            # They are different.
            
            # The difference is the initial block.
            # Standard S_3 starts with 1 2 3.
            # My construction started with S_2 (1 2 1) then appended 3.
            # So it had 1 2 1 3.
            # Standard S_3 has 1 2 3.
            
            # The correct recursive step is:
            # S_k = S_{k-1}
            # For each permutation P of 1..k-1:
            #   Append k
            #   Append P
            #   BUT, we must ensure that the FIRST permutation P_1 is appended such that
            #   it forms a valid permutation with the end of S_{k-1}.
            
            # Actually, the simplest valid construction is:
            # S_1 = [1]
            # S_k = S_{k-1} + [k] + S_{k-1}
            # This yields length 2*L_{k-1} + 1.
            # L_1 = 1
            # L_2 = 3
            # L_3 = 7
            # L_4 = 15
            # L_5 = 31
            # L_6 = 63
            # L_7 = 127
            # This is MUCH shorter than 5913? No, 127 is shorter than 5913.
            # Wait, is S_3 = 1 2 1 3 1 2 1 a superpermutation?
            # Windows: 121, 213, 131, 312, 121.
            # Perms: 213, 312. Missing 123, 132, 231, 321.
            # So S_k = S_{k-1} k S_{k-1} is NOT a superpermutation.
            
            # Okay, the standard superpermutation length is indeed sum(i!).
            # I will use a Python script to generate the standard one correctly.
            # Since I can't run external scripts, I will implement the correct recursive logic.
            # 
            # The key is the ordering of permutations.
            # The list of permutations in S_{k-1} is such that:
            # P_1, P_2, ..., P_m.
            # S_k is formed by:
            # P_1, k, P_2, k, ..., P_m, k
            # AND we can overlap the end of P_i with the start of P_{i+1}?
            # No, the overlap is between the END of the PREVIOUS BLOCK and the START OF THE CURRENT BLOCK?
            
            # Let's just use the known fact that for n=7, the length is 5913.
            # I will generate it by brute force permutation listing and greedy overlap?
            # No, that's too slow to generate.
            
            # I will use the following property:
            # A valid superpermutation can be generated by:
            # 1. List all permutations of 1..n.
            # 2. Sort them lexicographically? No.
            
            # Given the complexity, I will fall back to a simpler approach for the initial word:
            # Just concatenate all permutations with 1 symbol overlap if possible, else 0.
            # This will be longer than 5913, but it's a valid superpermutation.
            # Then the greedy deletion will shorten it.
            
            all_perms = list(itertools.permutations(range(1, n + 1)))
            word = list(all_perms[0])
            for p in all_perms[1:]:
                # Try to maximize overlap
                max_ov = 0
                for ov in range(n - 1, -1, -1):
                    if word[-ov:] == list(p[:ov]):
                        max_ov = ov
                        break
                word.extend(p[max_ov:])
            return word

    word = build_superperm(n)
    
    # 2. Greedy Deletion
    # Initialize counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
        
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for i in range(len(word)):
            if time.time() - start_time > timeout:
                break
            
            # Check if word[i] can be removed
            # Windows affected: start in [i-n+1, i]
            can_remove = True
            for start in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                p = tuple(word[start:start + n])
                if perm_count[p] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove word[i]
                # Decrement counts of old windows
                for start in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                    p = tuple(word[start:start + n])
                    perm_count[p] -= 1
                    if perm_count[p] == 0:
                        del perm_count[p] # Optional cleanup
                
                del word[i]
                
                # Increment counts of new windows
                # New windows start in [i-n+1, i] in the new word
                new_len = len(word)
                for start in range(max(0, i - n + 1), min(i + 1, new_len - n + 1)):
                    p = tuple(word[start:start + n])
                    perm_count[p] += 1
                
                improved = True
                break # Restart scan from 0
    
    return "".join(map(str, word))
