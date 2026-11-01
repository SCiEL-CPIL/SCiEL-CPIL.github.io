"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Classic recursive construction
    # This generates a superpermutation of length sum_{i=1}^n i!
    # For n=7, this is 5913.
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        # Extract unique (m-1)-permutations from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # The standard insertion pattern: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of 'new' and start of 'block'
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Efficient polish using coverage counts
    def build_coverage(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts
    
    improved = list(word)
    counts = build_coverage(improved, n)
    L = len(improved)
    
    def can_remove_block(start, length):
        """
        Checks if removing block [start, start+length) is safe.
        Returns list of affected windows if safe, else None.
        """
        if start < 0 or start + length > len(improved):
            return None
        
        affected = []
        # Windows that will be destroyed or shifted
        # A window at index j is affected if it overlaps with [start, start+length)
        # j <= start + length - 1 and j + n - 1 >= start
        # j >= start - n + 1 and j <= start + length - 1
        
        j_min = max(0, start - n + 1)
        j_max = min(start + length - 1, len(improved) - n)
        
        # We need to ensure that for every permutation P, 
        # if P was only present in the affected windows, it must still be present elsewhere.
        # Actually, simpler: 
        # 1. Collect all windows that overlap the removal zone.
        # 2. For each such window, check if its count > 1 OR if it is not a permutation (impossible, all windows are perms in a superperm? No, only if n=word_len. But generally windows are tuples of n distinct digits).
        # Wait, in a superpermutation, every window of length n is a permutation.
        # So we just need to ensure that no permutation is left with count 0.
        
        # Let's simulate the removal on the counts.
        # We need to know which windows disappear and which new windows appear (shift).
        # This is complex to do in-place. 
        # Alternative: 
        # The set of windows AFTER removal is:
        # - Windows entirely before start: unchanged.
        # - Windows entirely after start+length: unchanged, but indices shift.
        # - Windows overlapping the gap: 
        #   The windows that span the gap are formed by concatenating the prefix before the gap and the suffix after the gap.
        
        # Let's identify the specific windows that will be LOST (i.e., present before, not present after).
        # And windows that will be GAINED (present after, not present before).
        # If a window is LOST and its count was 1, it's unsafe.
        # If a window is GAINED, it's fine (it might have been 0 before, now 1).
        
        # Which windows are LOST?
        # Any window that contains at least one symbol from the removed block AND does not correspond to a valid window in the new string.
        # Actually, any window that overlaps the removed block is potentially lost or changed.
        
        # Let's just check the counts of windows that overlap the removed block.
        # If a window W overlaps the removed block, and count(W) == 1, and W is not re-created by the shift, it's unsafe.
        
        # How to check if W is re-created?
        # This is getting complicated. Let's use a simpler sufficient condition:
        # A removal is safe if every window that overlaps the removed block has count > 1.
        # This is conservative but safe. The seed uses this.
        # To be more aggressive, we can check if the specific window is preserved.
        
        # Let's stick to the conservative check first, but apply it to blocks of size 1 and 2.
        
        for j in range(j_min, j_max + 1):
            w = tuple(improved[j:j+n])
            if counts[w] <= 1:
                return None
            affected.append(w)
        
        return affected

    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        changed = False
        
        # Try removing blocks of size 1 and 2
        # We iterate from left to right.
        # If we remove a block, we update counts and restart or continue carefully.
        # Given the cost, restarting the scan is safer.
        
        # Direction 1: Left to Right
        i = 0
        while i < len(improved):
            if time.time() > deadline:
                return "".join(map(str, improved))
            
            # Try size 2 first (larger savings per check)
            removed = False
            for block_size in [2, 1]:
                if i + block_size > len(improved):
                    continue
                affected = can_remove_block(i, block_size)
                if affected is not None:
                    # Safe to remove
                    # Update counts
                    for w in affected:
                        counts[w] -= 1
                    
                    # Remove from list
                    del improved[i : i + block_size]
                    
                    # Note: New windows are formed at the junction. 
                    # We don't explicitly add them to counts here, which is a flaw.
                    # The counts for windows *not* overlapping the removal are correct.
                    # The counts for windows *overlapping* the removal were decremented.
                    # But new windows appear at the junction!
                    # Example: A B [C D] E F -> A B E F. 
                    # Window A B C D E (lost). Window A B E F (new).
                    # If we don't add A B E F to counts, we might think it's count 0 (if it wasn't there before) 
                    # or we might lose track of it.
                    # However, if A B E F is a valid permutation, it must be covered.
                    # If it wasn't there before, its count was 0. Now it is 1.
                    # Our `counts` dict doesn't know about it.
                    # This breaks the invariant that `counts` reflects the current state.
                    
                    # To fix this, we would need to recalculate counts for the junction area.
                    # Or, simpler: just rebuild counts after a few removals? No, too slow.
                    
                    # Let's try a different approach for the polish:
                    # Instead of maintaining counts dynamically, we can just check validity on the fly for the local area?
                    # No, that's $O(n)$ per check, which is fine.
                    
                    # Let's restart the polish loop with a full count rebuild if we want to be safe, 
                    # but that's $O(N)$ per removal. With 5000 removals, that's $25 \times 10^6$ ops. Feasible.
                    
                    # Actually, let's just break out and rebuild counts to be safe and simple.
                    # It ensures correctness.
                    # We'll do this: Remove, Rebuild Counts, Break inner loop, Restart outer scan.
                    
                    changed = True
                    # Rebuild counts to maintain consistency
                    counts = build_coverage(improved, n)
                    i = 0 # Restart scan
                    break # Break block_size loop
            
            if changed:
                continue # Restart the while i < len(improved) loop
            
            i += 1
        
        if not changed:
            break # No more simple deletions possible
            
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
