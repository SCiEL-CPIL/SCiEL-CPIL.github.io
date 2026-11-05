import itertools
import time
import sys
import random

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) 
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _try_delete(S, idx, n, counts):
    """
    Try to delete character at position idx from S.
    Returns True if deletion is valid (no permutation lost), and updates counts in place.
    If invalid, does not modify S or counts.
    """
    # Windows that contain position idx: windows starting at j where j <= idx <= j+n-1
    # i.e., j in [idx-n+1, idx], but also j >= 0 and j <= len(S)-n
    j_start = max(0, idx - n + 1)
    j_end = min(idx, len(S) - n)
    
    # Check if any window containing idx has count 1
    for j in range(j_start, j_end + 1):
        p = tuple(S[j:j+n])
        if counts.get(p, 0) == 1:
            return False
    
    # Deletion is valid. Now update counts.
    # First, decrement counts for all windows that contained idx
    for j in range(j_start, j_end + 1):
        p = tuple(S[j:j+n])
        counts[p] -= 1
        if counts[p] == 0:
            del counts[p]
    
    # Remove the character
    S.pop(idx)
    
    # Now, new windows are formed that straddle the deletion point.
    # In the new string, position idx-1 is followed by what was at idx+1.
    # New windows: those starting at j where the window crosses the gap.
    # In the new string, the gap is at position idx (0-indexed in new string).
    # Windows that cross: start at j where j < idx <= j+n-1 in new string coords.
    # But we need to be careful: the new string is S[:idx] + S[idx+1:] (old coords)
    # In new string, positions 0..idx-1 are same as old, positions idx.. are old idx+1..
    # A window starting at j in new string covers new positions j..j+n-1.
    # It crosses the gap if j < idx <= j+n-1, i.e., j in [idx-n+1, idx-1].
    # But j must be >= 0 and j <= len(S)-n (new length).
    
    new_len = len(S)
    nj_start = max(0, idx - n + 1)
    nj_end = min(idx - 1, new_len - n)
    
    for j in range(nj_start, nj_end + 1):
        # Construct window from new S
        # New position k corresponds to old position k if k < idx, else k+1
        window = []
        for k in range(j, j+n):
            old_pos = k if k < idx else k + 1
            window.append(S[old_pos])
        p = tuple(window)
        counts[p] = counts.get(p, 0) + 1
    
    return True

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety
    
    if n <= 3:
        word = _standard_superperm(n)
        return "".join(map(str, word))
    
    # Generate standard superpermutation
    S = _standard_superperm(n)
    best_S = S[:]
    best_len = len(S)
    
    # Build initial counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    # Multi-pass deletion with random restarts
    random.seed(42)
    
    while time.time() < deadline:
        pass_made = False
        # Try multiple passes, each trying to delete as many chars as possible
        # Process from end to start to avoid index shifting issues
        for idx in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _try_delete(S, idx, n, counts):
                pass_made = True
                if len(S) < best_len:
                    best_S = S[:]
                    best_len = len(S)
        
        if not pass_made:
            # Stuck. Try a random swap to potentially unlock deletions.
            if len(S) > 1 and time.time() < deadline - 5:
                # Pick two random positions to swap
                i = random.randint(0, len(S) - 2)
                j = random.randint(i + 1, len(S) - 1)
                S[i], S[j] = S[j], S[i]
                
                # Rebuild counts
                counts = {}
                for k in range(len(S) - n + 1):
                    p = tuple(S[k:k+n])
                    counts[p] = counts.get(p, 0) + 1
                
                # Verify it's still a valid superpermutation
                if not _verify_superpermutation(S, n):
                    # Revert
                    S[i], S[j] = S[j], S[i]
                    counts = {}
                    for k in range(len(S) - n + 1):
                        p = tuple(S[k:k+n])
                        counts[p] = counts.get(p, 0) + 1
                    continue
                
                # Try to delete after swap
                for idx2 in range(len(S) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    if _try_delete(S, idx2, n, counts):
                        if len(S) < best_len:
                            best_S = S[:]
                            best_len = len(S)
        
        # If we've been stuck for a while, try a more aggressive restart
        # But for now, just continue with the current approach
    
    # Final verification
    if not _verify_superpermutation(best_S, n):
        best_S = _standard_superperm(n)
    
    return "".join(map(str, best_S))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
