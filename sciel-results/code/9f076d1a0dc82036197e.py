"""Superpermutation task: minimize the length of a word over {1..n} containing every n-permutation as a contiguous window."""
import time
import random
from itertools import permutations

def construct(n):
    t_start = time.time()
    DEADLINE = 118.0  # seconds, safety margin before 120s kill

    if n <= 1:
        return "1" if n == 1 else ""

    # For n <= 4, use the classic recursive construction (optimal and fast)
    if n <= 4:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
            word = new
        return "".join(map(str, word))

    # For n >= 5: use the classic recursive construction as baseline.
    # This gives length sum_{i=1}^{n} i! = 5916 for n=7.
    # Then attempt local search / deletion to reduce length.
    
    def classic_construct(n):
        """Classic recursive superpermutation construction."""
        word = [1]
        for m in range(2, n + 1):
            # Collect all distinct (m-1)-length permutations that appear as windows
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
            word = new
        return word

    sp = classic_construct(n)
    sp_len = len(sp)
    
    if time.time() - t_start > DEADLINE - 5:
        return "".join(map(str, sp))

    # --- Fast deletion: find positions where removing a symbol doesn't lose coverage ---
    # Key insight: if two consecutive windows (at positions i and i+1) have the same
    # permutation, we can potentially remove the middle symbol.
    # More generally, if window at position i equals window at position i+k,
    # we can try to remove symbols between them.
    
    def fast_deletions(word, n, max_time=30):
        """Fast local search: find deletable positions using window equality."""
        t0 = time.time()
        word = list(word)
        
        while time.time() - t0 < max_time:
            L = len(word)
            if L <= n:
                break
            
            # Build window hash map: for each position, store the window tuple
            # and track which permutations are covered
            windows = {}
            perm_positions = {}  # perm -> list of positions where it appears
            
            for i in range(L - n + 1):
                w = tuple(word[i:i+n])
                if len(set(w)) == n:
                    if w not in perm_positions:
                        perm_positions[w] = []
                    perm_positions[w].append(i)
            
            # Check if all permutations are covered
            all_perms = set(permutations(range(1, n + 1)))
            if set(perm_positions.keys()) != all_perms:
                break
            
            # Try to find deletable positions
            deleted = False
            for pos in range(1, L - 1):
                if time.time() - t0 > max_time:
                    return word
                
                # A position pos is deletable if every permutation that appears
                # in a window containing pos also appears in a window NOT containing pos
                # Windows containing pos: start in [pos-n+1, pos] ∩ [0, L-n]
                j_min = max(0, pos - n + 1)
                j_max = min(pos, L - n)
                
                can_delete = True
                for j in range(j_min, j_max + 1):
                    w = tuple(word[j:j+n])
                    if len(set(w)) != n:
                        continue
                    # Check if w appears at another position not containing pos
                    found_elsewhere = False
                    for k in perm_positions.get(w, []):
                        if k != j and (k > pos or k + n - 1 < pos):
                            found_elsewhere = True
                            break
                    if not found_elsewhere:
                        can_delete = False
                        break
                
                if can_delete:
                    del word[pos]
                    deleted = True
                    break  # Restart scan after deletion
        
            if not deleted:
                break
        
        return word

    # Apply fast deletion pass
    if time.time() - t_start < DEADLINE - 10:
        sp = fast_deletions(sp, n, max_time=40)
    
    if time.time() - t_start > DEADLINE - 3:
        return "".join(map(str, sp))

    # --- Second pass: try more aggressive deletions with different starting points ---
    if time.time() - t_start < DEADLINE - 5:
        sp2 = list(sp)
        # Try deleting from the right side
        sp2 = fast_deletions(sp2, n, max_time=30)
        if len(sp2) < len(sp):
            sp = sp2

    # Final deadline check
    if time.time() - t_start > DEADLINE - 2:
        return "".join(map(str, sp))

    return "".join(map(str, sp))
