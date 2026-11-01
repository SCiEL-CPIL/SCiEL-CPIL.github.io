import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    if n == 1:
        return "1"
    
    # Step 1: Generate the classic recursive superpermutation
    # This produces a word of length 1! + 2! + ... + n!
    # For n=7, length is 13194. This is valid.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique permutations of length m-1 from the current word
        # The current word is a superpermutation for m-1 symbols.
        # We need to find all (m-1)-permutations appearing in it.
        # Actually, the standard construction says: take the list of permutations of size m-1 in order of appearance in the superpermutation for m-1.
        # The code below extracts them from the generated word.
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Only consider windows that are permutations (all distinct)
            # In the superpermutation for m-1, all (m-1)-length windows that are permutations are distinct and cover all (m-1)! perms.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # The block to insert is p, m, p
            block = list(p) + [m] + list(p)
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Optimization via Greedy Deletion
    # Precompute all target permutations
    target_perms = set(permutations(range(1, n + 1)))
    
    def is_valid(w):
        """Check if word w contains all target permutations."""
        if len(w) < n:
            return False
        present = set()
        # We can short-circuit if we find all
        for i in range(len(w) - n + 1):
            present.add(tuple(w[i:i + n]))
            # Optional: early exit if all found? 
            # len(present) == len(target_perms) is O(1) check if we use a counter or set difference.
            # But set construction is fast.
        return target_perms.issubset(present)

    # Initial validation (should be True)
    if not is_valid(word):
        return "".join(map(str, word))
        
    # Greedy deletion loop
    # We try to remove characters one by one.
    # To be more efficient, we can maintain the set of present permutations and update it incrementally?
    # That is complex. Given n=7 and length ~13k, is_valid is O(L * n). 
    # L=13194, n=7. 13194 * 7 ~ 90k ops.
    # 13194 deletions * 90k ops ~ 1.2 billion ops. Too slow in Python.
    
    # Optimization: 
    # Instead of full validation on every step, we can use a smarter check.
    # Or, we can limit the number of full validations.
    # Actually, for n=7, the reduction from 13194 to ~6000 is significant.
    # Let's try to optimize is_valid.
    
    # Better approach for is_valid:
    # We only need to check if *all* are present.
    # We can maintain a count of each permutation.
    # But implementing incremental update is risky for bugs.
    
    # Let's stick to full validation but optimize the inner loop.
    # Using a set of tuples is the standard way.
    
    # Alternative: 
    # Only try deletion if the character is "redundant" in some local sense?
    # No, global redundancy is required.
    
    # Let's estimate speed.
    # 13000 steps. If each step takes 1ms, total 13s. This is feasible.
    # Python tuple slicing and hashing is fast.
    
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        # Scan from left to right
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Try removing word[i]
            # We can skip if word[i] is the only occurrence of some permutation?
            # That's hard to know without full check.
            
            temp_word = word[:i] + word[i+1:]
            if is_valid(temp_word):
                word = temp_word
                changed = True
                break # Restart scan as indices changed
    
    return "".join(map(str, word))
