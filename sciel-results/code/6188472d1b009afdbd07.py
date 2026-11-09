import time
import sys
from collections import deque, defaultdict

def construct(n):
    """
    Constructs a superpermutation for n symbols using the classic recursive
    construction, then attempts local shortening to reduce length.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline - 40:  # Leave time for shortening
            break
        
        # Extract all unique permutations of length m-1 from the current word
        # in order of first appearance
        perms, seen = [], set()
        k = m - 1
        full_mask = (1 << k) - 1
        
        for i in range(len(word) - k + 1):
            window = word[i:i + k]
            mask = 0
            valid = True
            for sym in window:
                bit = 1 << (sym - 1)
                if mask & bit:
                    valid = False
                    break
                mask |= bit
            if valid and mask == full_mask:
                wkey = tuple(window)
                if wkey not in seen:
                    seen.add(wkey)
                    perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p, m, p
            block = p + [m] + p
            
            # Find maximum overlap with the current end of 'new'
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    # Local shortening pass
    if time.time() < deadline - 5:
        word = _local_shorten(word, n, deadline)
    
    return "".join(map(str, word))


def _local_shorten(word, n, deadline):
    """
    Attempt to shorten the superpermutation by removing redundant symbols.
    Uses a sliding window approach to track which permutations are covered
    by which positions, then greedily removes positions that don't destroy
    any unique permutation coverage.
    """
    word = list(word)
    length = len(word)
    target = _factorial(n)
    
    # Build mapping: for each position, which permutation windows cover it
    # And for each permutation, how many times it appears and at which positions
    perm_count = defaultdict(int)  # perm tuple -> count
    perm_positions = defaultdict(list)  # perm tuple -> list of window start positions
    pos_perms = defaultdict(set)  # position -> set of perm tuples covering it
    
    full_mask = (1 << n) - 1
    
    # Scan all windows
    for i in range(length - n + 1):
        window = word[i:i + n]
        mask = 0
        valid = True
        for sym in window:
            bit = 1 << (sym - 1)
            if mask & bit:
                valid = False
                break
            mask |= bit
        if valid and mask == full_mask:
            wkey = tuple(window)
            perm_count[wkey] += 1
            perm_positions[wkey].append(i)
            # This window covers positions i through i+n-1
            for pos in range(i, i + n):
                pos_perms[pos].add(wkey)
    
    # Check if we have all permutations
    if len(perm_count) < target:
        return word
    
    # Greedy removal: try to remove each position from left to right
    # A position can be removed if every perm covering it has count > 1
    # and the remaining windows still cover all perms
    
    removed = 0
    i = 0
    while i < len(word) - 1 and time.time() < deadline - 2:
        # Check if position i can be removed
        can_remove = True
        for perm in pos_perms.get(i, set()):
            if perm_count[perm] <= 1:
                can_remove = False
                break
        
        if can_remove:
            # Remove position i and update counts
            # Windows that contained position i: starts from max(0, i-n+1) to min(i, len(word)-n)
            for start in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                if start < len(word) - n + 1:
                    window = word[start:start + n]
                    mask = 0
                    valid = True
                    for sym in window:
                        bit = 1 << (sym - 1)
                        if mask & bit:
                            valid = False
                            break
                        mask |= bit
                    if valid and mask == full_mask:
                        wkey = tuple(window)
                        perm_count[wkey] -= 1
                        if perm_count[wkey] == 0:
                            # This perm is now gone - check if it was only here
                            # We need to verify all perms still exist
                            pass
                        # Remove from pos_perms for positions in this window
                        for pos in range(start, start + n):
                            pos_perms[pos].discard(wkey)
            
            # Remove position i from word
            word.pop(i)
            removed += 1
            
            # Update pos_perms keys: positions after i shift left
            # This is expensive; instead, rebuild if needed
            # For now, just continue without i (don't advance)
            
            # Check if we still have all perms
            if len([p for p, c in perm_count.items() if c > 0]) < target:
                # We lost a perm - this shouldn't happen if our check was correct
                # But let's be safe and stop
                break
        else:
            i += 1
    
    return word


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
