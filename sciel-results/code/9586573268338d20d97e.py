import time
import random
import itertools
import sys

def _standard_superperm(n):
    """Standard recursive superpermutation construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    word = _standard_superperm(n - 1)
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find max overlap between end of new_word and start of block
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def _find_superperm_5(time_limit=15):
    """Find a short superpermutation for n=5 using greedy random restarts."""
    n = 5
    symbols = list(range(1, n + 1))
    perms = [tuple(p) for p in itertools.permutations(symbols)]
    perm_set = set(perms)
    best_len = float('inf')
    best_word = None
    start_time = time.time()
    
    def greedy_start(start_perm):
        word = list(start_perm)
        seen = set()
        # Add perms in current word
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i+n]))
        remaining = perm_set - seen
        if not remaining:
            return word
            
        while remaining:
            # Find the permutation with maximum overlap
            best_overlap = -1
            best_next = None
            # For speed, iterate over a random subset if remaining is large
            if len(remaining) > 50:
                candidates = random.sample(list(remaining), 50)
            else:
                candidates = list(remaining)
                
            for p in candidates:
                p_list = list(p)
                max_ov = 0
                # Check overlaps from largest to smallest
                for ov in range(min(len(word), len(p_list)), 0, -1):
                    if word[-ov:] == p_list[:ov]:
                        max_ov = ov
                        break
                if max_ov > best_overlap:
                    best_overlap = ov
                    best_next = p
            if best_next is None:
                best_next = random.choice(list(remaining))
                best_overlap = 0
                
            # Append best_next with overlap
            word.extend(best_next[best_overlap:])
            remaining.remove(best_next)
            # Update seen
            # Only need to check the new windows
            start_idx = len(word) - n
            for i in range(start_idx, len(word) - n + 1):
                if i >= 0:
                    seen.add(tuple(word[i:i+n]))
            # Remove newly seen from remaining
            for s in list(remaining):
                if s in seen:
                    remaining.remove(s)
                    
        return word

    # Try a few deterministic starts first?
    # Or just random restarts.
    while time.time() - start_time < time_limit:
        start_perm = tuple(random.sample(symbols, n))
        word = greedy_start(start_perm)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            if best_len == 152: # Known optimal
                break
                
    if best_word is None:
        # Fallback to standard
        return _standard_superperm(5)
        
    return best_word

def _extend_superperm(word, m):
    """Extend a superpermutation of 1..m-1 to 1..m."""
    n = m
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find max overlap between end of new_word and start of block
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def _greedy_superperm(n, time_limit=10):
    """
    Greedy construction for n=6 or n=7.
    We build a word that contains all n! permutations as contiguous substrings.
    We use a priority queue of permutations based on overlap with the current suffix.
    """
    symbols = list(range(1, n + 1))
    perms = [tuple(p) for p in itertools.permutations(symbols)]
    perm_set = set(perms)
    
    # We will maintain the current word and a set of seen permutations.
    # To be efficient, we don't store the whole word in a list if it gets huge? 
    # For n=7, length is ~5000-6000. List is fine.
    
    best_len = float('inf')
    best_word = None
    start_time = time.time()
    
    # Initial random restarts
    # For n=7, pure random restarts with greedy might be too slow to find a good one in 150s?
    # The standard recursive gives 5912. 
    # A greedy approach on n=7 might be feasible if we are smart.
    
    # Let's try a greedy approach that picks the next permutation to maximize overlap.
    # To avoid O(N^2) or worse, we can use a heuristic.
    
    # Actually, for n=7, the search space is huge (5040 perms).
    # A simple greedy that picks the best overlap among ALL remaining perms is O(N^2 * N) = O(N^3) which is 5040^3 ~ 10^11. Too slow.
    # We need to limit candidates or use a different strategy.
    
    # Strategy:
    # 1. Start with a random perm.
    # 2. At each step, look at a random sample of remaining perms (e.g. 100-200) and pick the one with max overlap.
    # 3. If we get stuck (no overlap > 0), we might have to append with 0 overlap.
    
    # Let's try this.
    
    def greedy_once():
        word = list(random.sample(symbols, n))
        seen = set([tuple(word)])
        remaining = perm_set - seen
        
        while remaining:
            # Sample candidates
            if len(remaining) > 100:
                candidates = random.sample(list(remaining), 100)
            else:
                candidates = list(remaining)
            
            best_ov = -1
            best_p = None
            
            for p in candidates:
                # Check overlap
                # Max possible overlap is n-1
                for ov in range(n - 1, 0, -1):
                    if word[-ov:] == list(p[:ov]):
                        if ov > best_ov:
                            best_ov = ov
                            best_p = p
                        break
                # If no overlap found in the loop, best_ov for this p is 0
                # But we need to track if we found ANY overlap > 0
                # If best_ov is still -1, it means no overlap > 0 was found for ANY candidate?
                # No, best_ov is global.
                # We should track the best overlap found in this batch.
                
            # If no candidate had overlap > 0, pick a random one
            if best_p is None:
                best_p = random.choice(list(remaining))
                best_ov = 0
            
            # Append
            word.extend(best_p[best_ov:])
            remaining.remove(best_p)
            
            # Update seen
            # New windows are at the end
            start_idx = len(word) - n
            for i in range(start_idx, len(word) - n + 1):
                if i >= 0:
                    w = tuple(word[i:i+n])
                    if w in remaining:
                        remaining.remove(w)
                    # We don't strictly need to track 'seen' if we just remove from remaining
                    # But we need to ensure we don't pick a perm that is already in the word but not in 'remaining'?
                    # 'remaining' starts as all perms. We remove the one we just added.
                    # We also remove any new perms formed by the new suffix.
                    # This ensures 'remaining' only contains perms NOT yet in the word.
                    # So we don't need a separate 'seen' set, just 'remaining'.
                    
        return word

    # Run greedy for a while
    # For n=6, this is fast. For n=7, it might be slow per restart, but we can do many.
    # Let's estimate time per restart for n=7.
    # ~5040 steps. Each step samples 100 candidates, checks overlap (up to 6 comparisons).
    # 5040 * 100 * 6 = ~3 million ops. Python can do ~10-50M ops/sec. So ~0.1-0.3s per restart.
    # In 100s, we can do ~300-1000 restarts.
    
    while time.time() - start_time < time_limit:
        word = greedy_once()
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            # Print progress if debugging
            # print(f"New best: {best_len}")
            
    if best_word is None:
        return _standard_superperm(n)
        
    return best_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n <= 4:
        return "".join(map(str, _standard_superperm(n)))
        
    if n == 5:
        # Use the specific optimized finder for n=5
        word = _find_superperm_5(10)
        return "".join(map(str, word))
        
    if n == 6 or n == 7:
        # Use the general greedy finder
        # Allocate most of the time to this
        time_left = deadline - time.time() - 5.0
        if time_left < 10:
            time_left = 10
        word = _greedy_superperm(n, time_limit=time_left)
        return "".join(map(str, word))
        
    # Fallback for n > 7
    word = _standard_superperm(n)
    return "".join(map(str, word))

if __name__ == "__main__":
    # Quick test
    print(construct(3))
    print(len(construct(3)))
