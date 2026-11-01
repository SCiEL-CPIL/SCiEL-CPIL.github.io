import time
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Standard Recursive Construction
    # Length: sum_{k=1}^n k!
    # For n=7: 5913
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        # The current word is a superpermutation of 1..m-1
        # We need to order these permutations to form the superpermutation of 1..m
        # Standard method: 
        # For each permutation p of 1..m-1 in the order they appear in the word,
        # append p + [m] + p
        
        # Wait, the standard recursive step is:
        # S_m = S_{m-1} with m inserted.
        # Actually, the standard construction is:
        # S_1 = 1
        # S_{n} = S_{n-1} (with n appended to each permutation block?)
        
        # Let's use the explicit generation which is safer.
        # S_n is formed by taking S_{n-1}, and for every permutation of 1..n-1,
        # we insert n.
        
        # A simpler recursive definition:
        # Let P_{n-1} be the list of permutations of 1..n-1 in the order they appear in S_{n-1}.
        # S_n is constructed by:
        # Start with S_{n-1}.
        # For each permutation p in P_{n-1}:
        #    Replace p in S_{n-1} with p + [n] + p ? No.
        
        # Correct Standard Construction:
        # S_n is a superpermutation of 1..n.
        # It is constructed from S_{n-1} as follows:
        # S_{n-1} contains all permutations of 1..n-1.
        # Let the sequence of permutations of 1..n-1 appearing in S_{n-1} be \pi_1, \pi_2, ..., \pi_{(n-1)!}.
        # Then S_n is formed by:
        # \pi_1, n, \pi_1, \pi_2, n, \pi_2, ... \pi_{(n-1)!}, n, \pi_{(n-1)!}
        # Wait, this is not quite right. The overlap is key.
        
        # Let's just generate the list of permutations and use the known pattern.
        # The pattern is:
        # S_n = S_{n-1} with the last symbol of each (n-1)-block replaced by n, 
        # and then the (n-1)-block repeated?
        
        # Actually, the simplest way to generate the standard superperm is:
        # 1. Get all permutations of 1..n in lexicographic order? No, the order matters for length.
        # The "standard" superpermutation is the one that minimizes length among the recursive constructions.
        
        # Let's use the algorithm:
        # def superperm(n):
        #   if n == 1: return [1]
        #   prev = superperm(n-1)
        #   # Extract the sequence of (n-1)-permutations from prev
        #   perms = []
        #   for i in range(len(prev) - (n-1) + 1):
        #      w = tuple(prev[i:i+n-1])
        #      if len(set(w)) == n-1:
        #         perms.append(w)
        #   # Now construct new word
        #   new_word = []
        #   for p in perms:
        #      block = list(p) + [n] + list(p)
        #      # Append with max overlap
        #      # ...
        
        # This works. Let's implement it.
        
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word

    # 2. Optimization: Greedy Deletion
    # Target: Remove redundant symbols.
    
    # Precompute permutation indices for fast checking
    target_count = math.factorial(n)
    
    # Map permutation tuple to index
    # We need a fast way to check if a window is a permutation and get its index.
    # For n=7, we can just use a set of tuples for the "present" permutations.
    
    def get_perm_set(w):
        s = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                s.add(window)
        return s
    
    current_word = word
    current_perms = get_perm_set(current_word)
    current_len = len(current_word)
    
    # Iterate until no more deletions or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to delete from left to right
        # We need to be careful with indices.
        # If we delete index i, indices > i shift.
        # So we scan 0 to len-1.
        
        for i in range(len(current_word)):
            if time.time() > deadline:
                break
            
            # Check if current_word[i] is redundant
            # A symbol is redundant if removing it doesn't remove any unique permutation.
            # We can check this by:
            # 1. Identify all permutations in windows containing index i.
            # 2. Remove them from a copy of current_perms.
            # 3. Simulate removal.
            # 4. Add back permutations in the new windows formed by the removal.
            # 5. If the set size is still target_count, the removal is valid.
            
            # Windows containing i in current_word:
            # Start j such that j <= i <= j+n-1 => i-n+1 <= j <= i
            # j must be >= 0 and <= len-n
            
            start_j = max(0, i - n + 1)
            end_j = min(i, len(current_word) - n)
            
            removed_perms = set()
            for j in range(start_j, end_j + 1):
                window = tuple(current_word[j:j+n])
                if len(set(window)) == n:
                    removed_perms.add(window)
            
            # Check if any of these permutations are unique
            # i.e., if count(window) == 1
            # We don't have counts, just a set.
            # So we need to know if they appear elsewhere.
            
            # Efficient check:
            # If we remove index i, the new word is W'.
            # The windows in W' that are different from W are those that span the gap.
            # These are windows starting at j' in W' such that j' <= i-1 <= j'+n-1 (in W' coords)
            # Actually, it's easier to just construct the new set of permutations for the affected region.
            
            # Let's just do a full check for simplicity first, as n=7 is small.
            # But full check is O(L * n). L~5900, n=7. ~40k ops.
            # If we do this for each of 5900 positions, that's 240M ops. Too slow.
            
            # We need the local check.
            # A permutation P is lost if it was ONLY in the windows containing i.
            # So, for each P in removed_perms, check if P appears in any window NOT containing i.
            
            is_redundant = True
            for p in removed_perms:
                # Check if p appears in a window not containing i
                found_elsewhere = False
                for j in range(len(current_word) - n + 1):
                    if start_j <= j <= end_j:
                        continue # This window contains i, skip
                    if tuple(current_word[j:j+n]) == p:
                        found_elsewhere = True
                        break
                if not found_elsewhere:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Perform deletion
                current_word = current_word[:i] + current_word[i+1:]
                improved = True
                # Break and restart scan? Or continue?
                # If we continue, indices shift.
                # It's safer to break and restart the outer loop to rescan from 0.
                # Or just break the inner loop and let the while loop restart.
                break
                
    return "".join(map(str, current_word))
