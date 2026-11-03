"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _classic_superperm(n):
    """Build the classic recursive superpermutation of length sum(k! for k=1..n)."""
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

class _SuperpermState:
    """
    Maintains the count of occurrences of each permutation in a word.
    Allows efficient checking and updating for deletions.
    """
    def __init__(self, word, n):
        self.n = n
        self.word = word[:]
        self.length = len(self.word)
        self.perm_counts = {}
        self.total_unique = 0
        self.expected = math.factorial(n)
        
        # Initialize counts
        for i in range(self.length - n + 1):
            w = tuple(self.word[i:i+n])
            if len(set(w)) == n:
                if w in self.perm_counts:
                    self.perm_counts[w] += 1
                else:
                    self.perm_counts[w] = 1
                    self.total_unique += 1
                    
    def is_valid(self):
        return self.total_unique == self.expected

    def try_delete(self, i):
        """
        Try to delete element at index i.
        Returns True if successful, False otherwise.
        """
        n = self.n
        L = self.length
        
        if i <= 0 or i >= L - 1:
            return False
            
        # Windows that will be removed (start indices k such that window k..k+n-1 contains i)
        # i is in window starting at k if k <= i <= k+n-1  =>  i-n+1 <= k <= i
        # Also k must be valid start: 0 <= k <= L-n
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        
        removed_windows = []
        for k in range(start_min, start_max + 1):
            w = tuple(self.word[k:k+n])
            if len(set(w)) == n:
                removed_windows.append(w)
                # Check if this is the last occurrence
                if self.perm_counts[w] == 1:
                    # Cannot delete, would lose this permutation
                    return False
        
        # If we reach here, deletion is safe.
        # Perform deletion
        new_word = self.word[:i] + self.word[i+1:]
        
        # Update counts
        # 1. Decrement counts for removed windows
        for w in removed_windows:
            self.perm_counts[w] -= 1
            if self.perm_counts[w] == 0:
                del self.perm_counts[w]
                self.total_unique -= 1
        
        # 2. Add new windows formed by bridging the gap
        # New windows start at k in [i-n+1, i-1] (relative to new word indices? No, let's trace carefully)
        # Old word indices: ... i-1, i, i+1 ...
        # New word indices: ... i-1, i+1 ... (i+1 becomes i)
        # A new window in new word starting at k_new consists of:
        # new_word[k_new : k_new+n]
        # If k_new < i, the window might span the deletion point.
        # The windows that change are those that include the position where i was.
        # In the new word, the gap is at index i (elements after i shift left).
        # Windows in new word starting at k_new such that they cover the "seam" at i.
        # The seam is between new_word[i-1] and new_word[i] (which is old_word[i+1]).
        # So windows starting at k_new where k_new <= i-1 < k_new+n.
        # i.e., i-n+1 <= k_new <= i-1.
        # Also k_new must be valid: 0 <= k_new <= new_len - n.
        
        new_len = L - 1
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, new_len - n)
        
        for k_new in range(new_start_min, new_start_max + 1):
            # Construct the new window
            # It consists of new_word[k_new : i] + new_word[i : k_new+n]
            # Which is old_word[k_new : i] + old_word[i+1 : k_new+n+1]
            left_part = self.word[k_new:i]
            right_part = self.word[i+1:k_new+n+1]
            w_new = tuple(left_part + right_part)
            
            if len(set(w_new)) == n:
                if w_new not in self.perm_counts:
                    self.perm_counts[w_new] = 1
                    self.total_unique += 1
                else:
                    self.perm_counts[w_new] += 1
        
        # Update state
        self.word = new_word
        self.length = new_len
        return True

def _try_remove_chars_fast(word, n, time_limit):
    """Try to remove redundant characters from word while maintaining validity using incremental state."""
    start_time = time.time()
    state = _SuperpermState(word, n)
    
    # We want to remove characters.
    # Strategy: iterate from right to left, try to delete.
    # If successful, continue from right to left? Or restart?
    # Restarting is safer but slower.
    # Let's try a single pass from right to left first.
    # Then maybe another pass?
    
    # Actually, deleting a character shifts indices.
    # If we delete at i, characters at > i shift left.
    # So iterating from right to left is natural: we don't disturb indices < i.
    
    i = state.length - 1
    while i > 0:
        if (time.time() - start_time) >= time_limit:
            break
        
        if state.try_delete(i):
            # Deletion successful.
            # i now points to the element that was at i+1 (shifted left).
            # Wait, try_delete updates self.word and self.length.
            # The element at index i in the NEW word is the one that was at i+1 in the OLD word.
            # So we can just decrement i to check the next one to the left?
            # No, the element at i in the new word is the one we haven't checked yet (it was at i+1).
            # So we should NOT decrement i, we should check the same index i again?
            # Or rather, the index i in the new word corresponds to the old index i+1.
            # We want to check indices from right to left.
            # So if we delete at i, the next candidate to the left is i-1.
            # But the element at i in the new word is the one we just shifted into place.
            # Should we check it?
            # If we iterate i from L-1 down to 1:
            # 1. Check i.
            # 2. If deleted, length decreases by 1.
            # 3. The element now at i is the one that was at i+1.
            # 4. We haven't checked that element yet (we were moving right to left).
            # 5. So we should check index i again?
            #    No, that would be checking the same "logical position" from the right.
            #    Let's trace:
            #    Word: A B C D
            #    i=3 (D). Try delete D. Success. Word: A B C.
            #    Next i=2 (C). Try delete C. Success. Word: A B.
            #    Next i=1 (B). Try delete B. Success. Word: A.
            #    This works.
            #    What if we delete at i=2 (C) in A B C D?
            #    Word: A B D.
            #    Next i=1 (B).
            #    We skipped checking D?
            #    In the first case, we checked D at i=3.
            #    In the second case, if we delete C at i=2, D shifts to i=2.
            #    We move to i=1. We never check D in its new position?
            #    But D was at the end. If we move right to left, we check the rightmost first.
            #    So D (at i=3) is checked before C (at i=2).
            #    If we delete C, D is now at i=2.
            #    We move to i=1.
            #    We effectively "skipped" D's new position, but we already checked its old position.
            #    Is it possible that D becomes deletable only after C is removed?
            #    Yes.
            #    So a single pass might not be optimal.
            #    However, for time efficiency, a single pass or a few passes is good.
            #    Let's do a single pass first.
            i -= 1
        else:
            i -= 1
            
    return state.word

def construct(n):
    # Use deadline guard
    start_time = time.time()
    time_limit = 100  # Leave 20s safety margin for 120s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Try to remove redundant characters
    word = _try_remove_chars_fast(word, n, time_limit - (time.time() - start_time))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
