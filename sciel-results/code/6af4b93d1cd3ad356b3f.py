import time
import itertools

def construct(n):
    # Generate standard recursive superpermutation
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
            # Find overlap
            # The seed program logic:
            # overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            # This logic seems to assume the block starts with the previous permutation?
            # Let's use the standard construction directly.
            pass
            
    # Standard construction implementation
    def get_superperm(n):
        if n == 1:
            return [1]
        prev = get_superperm(n - 1)
        # Extract permutations of 1..n-1 in order of appearance
        perms, seen = [], set()
        for i in range(len(prev) - (n - 1) + 1):
            w = tuple(prev[i:i + n - 1])
            if len(set(w)) == n - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        
        new_word = []
        for i, p in enumerate(perms):
            # Append p
            new_word.extend(p)
            # Append n
            new_word.append(n)
            # If not last, append next perm?
            # The standard construction is:
            # S_n = P_1 n P_2 n P_3 ... n P_k
            # Where P_i are the perms.
            # But we need to handle the overlap.
            # Actually, the word is just concatenation of P_i n?
            # No, P_i and P_{i+1} are different.
            # The word is P_1 n P_2 n ... n P_k.
            # Let's check length.
            # k = (n-1)!
            # Length = k * (n-1) + k * 1 = k * n = n!
            # But L_n = n! + (n-1)! + ...
            # So this construction gives length n! only.
            # This is not a superpermutation.
            
        # The correct recursive construction:
        # S_n is formed by taking S_{n-1} and replacing each permutation P with P n P_next.
        # Where P_next is the next permutation in S_{n-1}.
        # This adds (n-1) + 1 + (n-1) - overlap?
        
        # Let's use the known valid construction:
        # S_n = S_{n-1} with each P replaced by P n P_{next}
        # But P_{next} is the next permutation in the sequence.
        # For the last permutation, P_{next} is the first permutation?
        # If we use the cyclic version, we get a cyclic superpermutation.
        # The linear version is obtained by breaking the cycle.
        
        # Let's just use the seed program's logic which is known to produce length 5913.
        return word

    # Re-implement seed logic correctly
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for i, p in enumerate(perms):
            block = list(p) + [m] + list(p)
            # Overlap with new
            # The seed program logic:
            # overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            # This finds the maximum overlap between the end of new and start of block.
            # This works if block starts with the previous permutation?
            # Let's trace.
            # If new ends with P_prev, and block starts with P_prev, then overlap is len(P_prev).
            # So we append m + P_prev?
            # No, block is P_prev m P_prev.
            # If new ends with P_prev, and block starts with P_prev, overlap is len(P_prev).
            # So we append m + P_prev.
            # This matches P_prev m P_prev.
            
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Now word is a list of ints.
    # Convert to string for easier manipulation?
    # Or keep as list.
    
    # Greedy removal
    # Generate all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def is_valid(w):
        # w is a list of ints
        if len(w) < n:
            return False
        windows = set()
        for i in range(len(w) - n + 1):
            windows.add(tuple(w[i:i + n]))
        return all_perms.issubset(windows)
    
    # Check initial validity
    if not is_valid(word):
        return "".join(map(str, word))
    
    # Try to remove characters
    # We will try to remove characters from index 0 to len(word)-1
    # If removal is valid, we remove it and restart the scan from the beginning.
    # To avoid infinite loops or too many restarts, we limit the number of restarts.
    # But since we only remove 8 chars, it should be fine.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    removed = 0
    while removed < 10 and time.time() < deadline:
        progress = False
        for i in range(len(word)):
            if time.time() > deadline:
                break
            # Try removing word[i]
            new_word = word[:i] + word[i+1:]
            if is_valid(new_word):
                word = new_word
                removed += 1
                progress = True
                # Restart scan from beginning
                break
        
        if not progress:
            break
    
    return "".join(map(str, word))
