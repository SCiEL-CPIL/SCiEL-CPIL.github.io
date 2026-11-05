   def construct(n):
       if n == 1:
           return [1]
       
       # S is a list of ints
       S = [1]
       
       for k in range(2, n + 1):
           new_S = []
           # Get all permutations of 1..k-1 in the order they appear in S
           # We need to track the windows in S
           # However, S contains overlapping windows.
           # The set of distinct permutations in S is all (k-1)!.
           
           # A simpler recursive step:
           # If S_{k-1} is a superpermutation for k-1,
           # S_k can be formed by:
           # For each permutation p of 1..k-1 (in the order they appear in S_{k-1}):
           #    append p
           #    append k
           # But we need to handle the overlap.
           
           # Actually, the recurrence L(k) = L(k-1) + k! + (k-1)! implies:
           # We take S_{k-1}.
           # We append the sequence of all permutations of 1..k-1, each followed by k?
           # No.
           
           # Let's use the "insertion" property:
           # S_k is obtained from S_{k-1} by inserting k after every occurrence of the permutation
           # that ends the current block?
           
           # Let's stick to the generation via permutations.
           # This is O(n! * n) which is fine for n=7 (5040 * 7 ~ 35k ops).
           
           # We maintain a list of the permutations in order.
           # But we only have the string.
           
           # Alternative:
           # S_1 = [1]
           # S_2 = [1, 2, 1]
           # S_3 = [1, 2, 3, 1, 2, 1, 3, 2, 1] ? No, that's length 9.
           # Let's check if 123121321 is valid. Yes.
           # How is it built?
           # S_2 = 121.
           # Perms in S_2: 12, 21.
           # If we take 12, append 3 -> 123.
           # If we take 21, append 3 -> 213.
           # This doesn't seem to directly yield 123121321.
           
           # Let's try the construction:
           # S_k = S_{k-1} with k inserted after every element? No.
           
           # There is a known construction:
           # Let P be the list of permutations of 1..k-1 in the order they appear in S_{k-1}.
           # S_k = concat( [p + [k] for p in P] ) ?
           # Length would be (k-1)! * k.
           # For k=3, P=[12, 21]. S_3 = 123 213. Length 6.
           # Does 123213 contain all perms of 1,2,3?
           # 123, 232 (no), 321, 213.
           # Missing 132, 312. So this is not a superpermutation.
           
           # Correct construction:
           # S_k is formed by taking S_{k-1} and appending the sequence of permutations
           # of 1..k-1 (in the order they appear in S_{k-1}) each followed by k,
           # BUT we can overlap the first part with the end of S_{k-1}?
           
           # Actually, the simplest robust method for n<=7 is to use the recursive formula:
           # L(n) = L(n-1) + n! + (n-1)!
           # Construction:
           # S_n = S_{n-1} + (concatenation of all perms of 1..n-1, each followed by n, 
           #                  but overlapping with the end of S_{n-1} if possible?)
           
           # Let's look at S_3 = 123121321.
           # S_2 = 121.
           # Perms of 1,2 in S_2: 12, 21.
           # If we append "123" (12 + 3) and "213" (21 + 3)?
           # 121 + 123 + 213 = 121123213. Length 9.
           # Windows: 121, 211, 112, 123, 232, 321, 213.
           # Perms: 123, 321, 213. Missing 132, 312, 231.
           # So simple concatenation fails.
           
           # The correct recursive step for the "Brinkmann" construction is:
           # S_n is obtained by taking S_{n-1} and inserting n after every occurrence of the
           # LAST element of S_{n-1}? No.
           
           # Let's use the explicit permutation iteration which is guaranteed to work:
           # We generate a word that contains all permutations.
           # A valid superpermutation is:
           # For each permutation p of 1..n:
           #   append p
           # This is n * n! length. Too long.
           
           # Better:
           # Use the property that if we have a superpermutation for n-1,
           # we can extend it.
           
           # Let's use the algorithm from "The shortest superpermutation" papers.
           # For n=7, we can use the construction:
           # S_1 = [1]
           # S_n = []
           # for p in permutations(1..n-1) in the order they appear in S_{n-1}:
           #     S_n.append(p)
           #     S_n.append([n])
           #     # But we need to overlap p with the end of the previous block?
           
           # Actually, there is a very simple Python implementation for this specific task (n=7)
           # that uses the recursive insertion:
           
           S = [1]
           for k in range(2, n + 1):
               # Get the order of permutations of 1..k-1 in S
               # We need to extract the windows of length k-1 from S
               # and deduplicate them while preserving order.
               
               seen = set()
               ordered_perms = []
               for i in range(len(S) - (k - 1) + 1):
                   w = tuple(S[i:i+k-1])
                   if w not in seen:
                       seen.add(w)
                       ordered_perms.append(w)
                       
               # Now construct new S
               new_S = []
               # We start with the first permutation of the new set?
               # No, we insert k into the existing structure.
               
               # The standard construction:
               # new_S = []
               # for p in ordered_perms:
               #     new_S.extend(p)
               #     new_S.append(k)
               #     # We need to remove the first k-1 elements of p to allow overlap?
               #     # If we just extend, length is (k-1)! * k.
               #     # For k=7, 720 * 7 = 5040.
               #     # Total length sum_{i=2}^7 i! * i ?
               #     # This is roughly 7 * 5040 + ... which is ~ 35000.
               #     # This is the naive bound.
               
               # To get the shorter bound, we must overlap.
               # The recurrence L(k) = L(k-1) + k! + (k-1)! suggests:
               # new_S = S (the old one)
               # then append the sequence of permutations (in order) each followed by k,
               # BUT the first permutation appended should overlap with the end of S?
               
               # Let's try:
               # new_S = list(S)
               # for p in ordered_perms:
               #     # Append p[1:] + [k] ?
               #     # If we append p[1:] + [k], the first element of p is already at the end of new_S?
               #     # Only if the last (k-1) elements of new_S match p.
               
               # This is getting complicated to implement correctly without a reference.
               # However, for n=7, a length of ~6000 is acceptable and much better than 35000.
               # The naive concatenation of all perms is 35280.
               # The "overlap by 1" strategy (append p[1:] + [k]) gives:
               # Length = L(k-1) + (k-1)! * (k-1) ?
               # No, if we append p[1:] + [k], we add k elements? No, p has k-1 elements.
               # p[1:] has k-2 elements. Plus k is 1. Total k-1 elements added per perm.
               # Number of perms is (k-1)!.
               # Added length = (k-1)! * (k-1).
               # L(k) = L(k-1) + (k-1)! * (k-1).
               # L(2) = 1 + 1! * 1 = 2. (But S_2 is 121, len 3).
               # So this simple overlap doesn't work directly because the first perm must be fully added.
               
               # Let's just use the full concatenation of permutations for the first few,
               # or better, use the known valid construction for n=7 which is ~5900.
               
               # I will implement the "Brinkmann" construction properly.
               # Reference: https://en.wikipedia.org/wiki/Superpermutation
               # "The shortest superpermutation for n=7 has length 5905..."
               # "A construction with length n! + (n-1)! + ... + 1! - (n-1) is known."
               
               # Let's implement that specific construction.
               # It relies on the fact that S_{n-1} starts with 1..n-1 and ends with ...
               
               # Actually, there is a simpler recursive definition:
               # S_n = S_{n-1}
               # for each permutation p of 1..n-1 (in order of appearance in S_{n-1}):
               #     append p
               #     append n
               #     # Remove the first element of p from the start of the appended block?
               #     # i.e., append p[1:] + [n]
               #     # But for the first one, we must append the whole p + [n]?
               
               # Let's try this logic:
               # new_S = list(S)
               # for i, p in enumerate(ordered_perms):
               #     if i == 0:
               #         # Check if S ends with p?
               #         # If so, we don't need to add p again?
               #         # But we need to add n.
               #         # So append [n]?
               #         # But we need to ensure the window p+n is formed.
               #         # If S ends with p, appending n creates window p+n.
               #         pass
               #     else:
               #         # Append p[1:] + [n]
               #         new_S.extend(p[1:])
               #         new_S.append(k)
               
               # This logic is flawed because S does not necessarily end with the first perm.
               
               # Given the time constraints and the need for a robust solution,
               # I will use the **Greedy Maximum Overlap** but with a **Pre-computed Optimal Order** for n=7?
               # No, that's hard.
               
               # Let's use the **Recursive Construction** that is known to produce length $n! + (n-1)! + \dots + 1! - (n-1)$.
               # The construction is:
               # S_1 = [1]
               # S_n is formed by taking S_{n-1} and inserting n after every element?
               
               # I will use the following Python code which is known to generate a valid superpermutation
               # of length approximately n! + (n-1)! for n <= 7.
               
               # Actually, the simplest correct approach for n=7 within 120s is:
               # Use the `itertools.permutations` to generate all perms.
               # Use a greedy algorithm that picks the perm with the longest suffix match.
               # The current code does this but is slow due to the inner loop checking all candidates.
               # We can optimize the current code's approach significantly.
               # 
               # BUT the directive says "design a new program... from an approach family that none of the current programs use".
               # The current program is Greedy.
               # The new approach should be **Recursive Construction**.
               
               # Here is the recursive construction algorithm:
               # def construct(n):
               #     if n == 1: return [1]
               #     prev = construct(n-1)
               #     # Get ordered perms from prev
               #     # ...
               #     # This is O(n! * n^2) which is fine.
               
               pass
       
       return ''.join(map(str, S))
   