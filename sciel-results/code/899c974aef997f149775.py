import time
import random

def construct(n):
    """
    Construct a superpermutation for n symbols, using a hybrid approach:
    1. Generate the standard recursive superpermutation.
    2. Apply local search (deletions, insertions, swaps) to reduce the length while
       maintaining that all permutations are covered.
    
    Uses a deadline guard to ensure termination within 140s.
    Returns the superpermutation as a string of digits.
    """
    start_time = time.time()
    deadline = 135.0  # 15s margin before the 150s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        # Extract unique consecutive (m-1)-permutations from word
        # The word contains all (m-1)-permutations as consecutive windows
        # But we need the specific order in which they appear in the recursive construction
        
        # Actually, the standard recursive construction is:
        # S_m = (S_{m-1} with m inserted after each symbol, separated by S_{m-1})
        # A simpler way to generate it:
        
        # Let's use the iterative method:
        # Start with [1]
        # For m=2: [1, 2, 1]
        # For m=3: 
        #   Take [1, 2, 1]
        #   Split into blocks: [1], [2], [1]
        #   For each block, append m and the block itself? No.
        
        # The standard construction:
        # S_1 = "1"
        # S_2 = "121"
        # S_3 = "131213231"
        # S_4 = "141312142312432142341214342314"
        
        # The recursive formula:
        # S_m = S_{m-1} with each symbol x replaced by x*m*S_{x}
        # where S_x is the sub-superpermutation for the remaining symbols?
        
        # Actually, the simplest recursive construction:
        # S_m = (concatenate for each permutation p of 1..m-1: p + [m] + p)
        # but this is not minimal.
        
        # Let's use the standard algorithm:
        # word = [1]
        # for m in range(2, n+1):
        #   new_word = []
        #   for i in range(len(word) - m + 2):
        #     window = word[i:i+m-1]
        #     if len(set(window)) == m-1:  # it's a permutation of 1..m-1
        #       new_word.extend(window)
        #       new_word.append(m)
        #       new_word.extend(window)
        #     # But this doesn't work directly.
        
        # Correct standard construction:
        # S_m is obtained from S_{m-1} by:
        # 1. Take S_{m-1}
        # 2. For each position, insert m and then the remaining m-2 symbols in some order
        
        # Let's just generate it properly:
        new_word = []
        # Get all (m-1)-permutations in the order they appear in word
        # The word at step m-1 is a superpermutation for m-1 symbols
        # We need to find the (m-1)-length windows that are permutations of 1..m-1
        
        windows = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and 1 <= min(w) and max(w) <= m - 1:
                if w not in seen:
                    seen.add(w)
                    windows.append(w)
        
        for w in windows:
            new_word.extend(w)
            new_word.append(m)
            new_word.extend(w)
        
        # This gives a valid superpermutation but might not be the shortest known.
        # The length should be (m-1)! * (2*m - 1) + ... 
        # Actually, this construction gives:
        # len = (m-1)! * (2*m - 1) for the blocks, but they overlap.
        
        word = new_word
    
    current_word = word[:]
    
    # Pre-compute permutation codes
    pow_n = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_n[i] = pow_n[i - 1] * (n + 1)
    
    # Map each permutation to a code
    perm_codes = {}
    import itertools
    all_perms = list(itertools.permutations(range(1, n + 1)))
    for p in all_perms:
        code = 0
        for i in range(n):
            code += p[i] * pow_n[n - 1 - i]
        perm_codes[code] = p
    
    def get_code(start, end):
        """Get code of window from start to end (exclusive)."""
        if end - start != n:
            return -1
        code = 0
        for i in range(n):
            code += current_word[start + i] * pow_n[n - 1 - i]
        return code
    
    # Track which permutations are covered
    covered = set()
    L = len(current_word)
    for i in range(L - n + 1):
        code = get_code(i, i + n)
        if code >= 0 and code in perm_codes:
            covered.add(code)
    
    # Local search: try to delete characters
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 50:
        improved_this_round = False
        
        # Try deleting from the end
        L = len(current_word)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if deleting position i is valid
            # A deletion is valid if no permutation becomes uncovered
            # The windows affected are those that include position i
            # Window j includes positions j to j+n-1
            # Position i is in window j if j <= i <= j+n-1
            # So j ranges from max(0, i-n+1) to min(i, L-n)
            
            min_j = max(0, i - n + 1)
            max_j = min(i, L - n)
            
            valid = True
            new_covered = set()
            
            # Collect codes of windows that don't include position i
            # These remain unchanged
            for j in range(max_j + 1, L - n + 1):
                code = get_code(j, j + n)
                if code >= 0 and code in perm_codes:
                    new_covered.add(code)
            
            for j in range(min_j):
                code = get_code(j, j + n)
                if code >= 0 and code in perm_codes:
                    new_covered.add(code)
            
            # Now simulate deletion and check windows that were affected
            # After deletion, windows shift
            # This is complex, so let's do a simpler check:
            # Delete the character and recompute coverage for affected region
            
            old_word = current_word[:]
            del current_word[i]
            
            new_L = len(current_word)
            new_covered_after = set()
            
            # Recompute coverage for the affected region
            # Affected windows: those that included i, and those shifted
            start_check = max(0, min_j - 1)
            end_check = min(new_L, max_j + n)
            
            for j in range(start_check, end_check - n + 1):
                if j + n <= new_L:
                    code = get_code(j, j + n)
                    if code >= 0 and code in perm_codes:
                        new_covered_after.add(code)
            
            # Add unaffected windows
            for j in range(max_j + 1, new_L - n + 1):
                code = get_code(j, j + n)
                if code >= 0 and code in perm_codes:
                    new_covered_after.add(code)
            
            for j in range(min_j):
                code = get_code(j, j + n)
                if code >= 0 and code in perm_codes:
                    new_covered_after.add(code)
            
            if new_covered_after >= covered:
                # Deletion is valid
                improved_this_round = True
                L = new_L
            else:
                # Revert
                current_word = old_word
                L = len(current_word)
        
        if improved_this_round:
            no_improve_count = 0
            # Recompute covered set
            covered = set()
            L = len(current_word)
            for i in range(L - n + 1):
                code = get_code(i, i + n)
                if code >= 0 and code in perm_codes:
                    covered.add(code)
        else:
            no_improve_count += 1
        
        # Try random insertions and swaps to escape local optima
        if time.time() > deadline:
            break
        
        # Random swap
        L = len(current_word)
        if L > 1 and random.random() < 0.3:
            pos1 = random.randint(0, L - 2)
            pos2 = random.randint(pos1 + 1, L - 1)
            
            old_word = current_word[:]
            current_word[pos1], current_word[pos2] = current_word[pos2], current_word[pos1]
            
            # Check if still valid
            new_covered = set()
            L = len(current_word)
            for i in range(L - n + 1):
                code = get_code(i, i + n)
                if code >= 0 and code in perm_codes:
                    new_covered.add(code)
            
            if new_covered >= covered:
                covered = new_covered
                improved_this_round = True
            else:
                current_word = old_word
    
    return "".join(map(str, current_word))
