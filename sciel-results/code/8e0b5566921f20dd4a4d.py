import time
import random
import sys

def construct_standard(n):
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def get_affected_windows(S, idx, n):
    """
    Returns a list of (start_index, tuple_window) for all windows 
    that include the character at S[idx].
    """
    starts = []
    min_start = max(0, idx - n + 1)
    max_start = min(idx, len(S) - n)
    for j in range(min_start, max_start + 1):
        starts.append(j)
    return starts

def construct(n):
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # 1. Generate standard
    S = construct_standard(n)
    
    # 2. Initialize counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        if len(set(p)) == n: # Only count valid permutations
            counts[p] = counts.get(p, 0) + 1
            
    # We must ensure all 5040 perms are present initially.
    # The standard construction guarantees this.
    
    best_S = S[:]
    best_len = len(S)
    
    def try_remove(S, counts, n):
        """
        Attempts to remove characters from S using greedy method.
        Returns (new_S, new_counts, removed_count)
        """
        removed = 0
        i = 0
        L = len(S)
        while i < L:
            # Check if S[i] can be removed
            # Affected windows are those starting from i-n+1 to i
            min_start = max(0, i - n + 1)
            max_start = min(i, L - n)
            
            can_remove = True
            # We need to check if any of these windows has count 1
            # Note: If we remove S[i], these windows disappear.
            # New windows will form, but they can only *increase* counts.
            # So we only care if we lose a unique permutation.
            
            for j in range(min_start, max_start + 1):
                w = tuple(S[j:j+n])
                if len(set(w)) == n:
                    if counts.get(w, 0) == 1:
                        can_remove = False
                        break
            
            if can_remove:
                # Perform removal
                L -= 1
                S.pop(i)
                removed += 1
                
                # Update counts
                # Old windows (that included i) are removed.
                # New windows (straddling the gap) are added.
                
                # 1. Decrement old windows
                # The indices of old windows were min_start to max_start.
                # After removal, the string is shorter.
                # The windows that *existed* and *included* i are gone.
                # We need to subtract their counts.
                for j in range(min_start, max_start + 1):
                    w = tuple(S[j:j+n]) # Wait, S is modified. 
                    # We need the *old* windows.
                    # Let's reconstruct old windows from the pre-removal state?
                    # It's easier to just recompute counts for the affected region or globally.
                    # Given L ~ 6000, global recompute is O(L*n) = 42k ops.
                    # If we do this for every removal, and we remove ~100 chars, 4.2M ops.
                    # This is fast enough in Python? Maybe.
                    # But we can be smarter.
                    pass
                
                # Let's use the global recompute strategy for simplicity and safety, 
                # but optimize it by only doing it if we actually removed something.
                new_counts = {}
                for k in range(len(S) - n + 1):
                    w = tuple(S[k:k+n])
                    if len(set(w)) == n:
                        new_counts[w] = new_counts.get(w, 0) + 1
                counts = new_counts
                
                # Do not increment i, check same index again
            else:
                i += 1
                
        return S, counts, removed

    # Initial Greedy Pass
    S, counts, rem = try_remove(S, counts, n)
    best_S = S[:]
    best_len = len(S)
    
    # 3. Local Search with Swaps
    # If stuck, try swaps.
    # A swap is useful if it enables a removal.
    
    stuck_passes = 0
    max_stuck_passes = 5 # If we try 5 times to unstick via swaps and fail, give up?
    
    while time.time() < deadline:
        # Try to remove more
        S, counts, rem = try_remove(S, counts, n)
        if rem > 0:
            best_S = S[:]
            best_len = len(S)
            stuck_passes = 0
            continue
        else:
            stuck_passes += 1
            if stuck_passes > 20: # Stuck for a while
                break
            
        # Try a random swap
        if len(S) < 2:
            break
            
        # Pick random i, j
        # To make it effective, pick i, j that are somewhat close? 
        # Or completely random?
        # Random is fine.
        
        for _ in range(10): # Try 10 random swaps per "stuck" phase
            if time.time() > deadline:
                break
                
            i = random.randint(0, len(S) - 1)
            j = random.randint(0, len(S) - 1)
            if i == j:
                continue
            
            # Save state
            old_S = S[:]
            old_counts = counts.copy()
            
            # Perform Swap
            S[i], S[j] = S[j], S[i]
            
            # Check Validity (Did we lose any permutation?)
            # We only need to check if any count drops to 0.
            # We can update counts for affected windows.
            # Affected windows are those covering i or j.
            
            valid = True
            # Get affected start indices
            affected_starts = set()
            for idx in [i, j]:
                min_start = max(0, idx - n + 1)
                max_start = min(idx, len(S) - n)
                for k in range(min_start, max_start + 1):
                    affected_starts.add(k)
            
            # We need to compare old and new windows.
            # It's safer to just recompute counts for the whole string?
            # No, that's O(L*n). 10 swaps * 42k = 420k ops. Fast.
            # Let's just recompute counts for the whole string to check validity.
            # If valid, we then try to remove.
            
            new_counts = {}
            for k in range(len(S) - n + 1):
                w = tuple(S[k:k+n])
                if len(set(w)) == n:
                    new_counts[w] = new_counts.get(w, 0) + 1
            
            # Check if all original permutations are still present
            # We can just check if len(new_counts) == len(old_counts) 
            # AND all keys in old_counts are in new_counts with count >= 1?
            # Actually, we just need to ensure we didn't lose any *unique* ones.
            # If a permutation was present in old_counts, it must be in new_counts.
            # If new_counts has a permutation not in old_counts, that's fine (we gained one).
            
            for p, c in old_counts.items():
                if new_counts.get(p, 0) == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                S = old_S
                counts = old_counts
                continue
            
            # If valid, try to remove
            # Use the new_counts
            S_new, counts_new, rem_new = try_remove(S, new_counts, n)
            
            if rem_new > 0:
                # Improvement!
                S = S_new
                counts = counts_new
                best_S = S[:]
                best_len = len(S)
                stuck_passes = 0
                break # Restart the outer loop logic
            else:
                # No improvement, revert swap
                S = old_S
                counts = old_counts

    return "".join(map(str, best_S))
