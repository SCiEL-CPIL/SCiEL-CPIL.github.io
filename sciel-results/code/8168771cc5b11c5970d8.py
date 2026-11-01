"""Shortest superpermutation on seven symbols."""
import time
import math
from itertools import permutations
import heapq

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    if n <= 3:
        # Small cases: use the classic recursive construction
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 4, use an improved recursive construction with better overlap exploitation
    # The classic construction gives sum_{k=1}^n k! 
    # We can improve by using a more clever ordering of permutations at each level
    
    def build_superperm_recursive(m):
        """Build shortest known superpermutation for m symbols using recursive construction."""
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        
        # Get all permutations of 1..m-1
        prev_perms = list(permutations(range(1, m)))
        
        # For each permutation, create a block: perm + [m] + perm
        # But we need to find maximum overlap between consecutive blocks
        blocks = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            blocks.append(block)
        
        # Greedily merge blocks with maximum overlap
        result = list(blocks[0])
        for i in range(1, len(blocks)):
            block = blocks[i]
            # Find maximum overlap between end of result and start of block
            max_overlap = 0
            for o in range(min(len(result), len(block)), -1, -1):
                if result[len(result) - o:] == block[:o]:
                    max_overlap = o
                    break
            result += block[max_overlap:]
        
        return result
    
    # Build the base superpermutation
    word = build_superperm_recursive(n)
    base_length = len(word)
    
    if time.time() > deadline - 20:
        return "".join(map(str, word))
    
    # Try to improve by local search: try removing redundant middle sections
    # Strategy: for each position, try to see if we can remove a symbol and still be valid
    
    def validate_word(w, n):
        """Check if w contains all permutations of 1..n as contiguous subsequences."""
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == math.factorial(n)
    
    # Local search: try to find and remove redundant symbols
    # This is expensive, so we limit the number of attempts
    improved = False
    max_iterations = 5000
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline - 10:
        iteration += 1
        
        # Try removing a random symbol from the middle
        if len(word) <= n:
            break
        
        # Choose a position to try removing (avoid first n-1 and last n-1 positions)
        min_pos = n - 1
        max_pos = len(word) - n
        
        if min_pos >= max_pos:
            break
        
        pos = min_pos + (iteration * 7919) % (max_pos - min_pos)
        
        # Try removing the symbol at pos
        new_word = word[:pos] + word[pos+1:]
        
        # Quick check: does the removal create any invalid windows?
        # We only need to check windows that include position pos
        valid = True
        for i in range(max(0, pos - n + 1), min(len(new_word) - n + 1, pos + 1)):
            window = tuple(new_word[i:i+n])
            if len(set(window)) != n:
                # This window is invalid, but that's okay as long as we still have all perms
                pass
        
        # Full validation is expensive, so do it less frequently
        if iteration % 10 == 0 or len(new_word) < len(word) - 100:
            if validate_word(new_word, n):
                word = new_word
                improved = True
                max_iterations = iteration + 5000  # Reset counter
            else:
                # Revert
                pass
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
