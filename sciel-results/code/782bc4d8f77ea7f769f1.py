# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    
    deadline = time.time() + 140  # Safety margin before 150s limit

    # 1. Generate standard superpermutation of length 1! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of 'new'
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # For n <= 6, the standard construction is often optimal or close to it, 
    # and local search might be overkill or risky if time is tight, 
    # but for n=7 we definitely want to try to improve.
    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts of each permutation in the current word
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    def is_valid_deletion(i):
        """Check if deleting word[i] maintains validity. Returns list of changes if valid, else None."""
        L = len(word)
        start_k = max(0, i - n + 1)
        end_k = min(i, L - n)
        
        changes = []
        
        # 1. Decrement counts for old windows
        for k in range(start_k, end_k + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                changes.append((pid, -1))
                if counts[pid] < 0:
                    # Revert
                    for cid, delta in changes:
                        counts[cid] -= delta
                    return None
                if counts[pid] == 0:
                    # Mark as at risk, but don't fail yet
            
        # 2. Increment counts for new windows
        new_start_k = max(0, i - n)
        new_end_k = min(i - 1, L - n - 1)
        
        for k_prime in range(new_start_k, new_end_k + 1):
            w1 = word[k_prime:i]
            w2 = word[i+1:k_prime+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                changes.append((pid, 1))
        
        # Check if any count is 0
        for pid, _ in changes:
            if counts[pid] == 0:
                # Revert
                for cid, delta in changes:
                    counts[cid] -= delta
                return None
                
        return changes

    def apply_deletion(i, changes):
        word.pop(i)
        # Changes already applied to counts

    def revert_deletion(i, changes):
        # Note: word is already popped? No, we check first.
        # If we return None, we don't pop.
        # If we return changes, we pop.
        pass

    def try_swap(i, j):
        """Try swapping word[i] and word[j]. Returns True if valid and applied."""
        if i == j:
            return False
        if abs(i - j) < 2:
            return False # Adjacent swaps are handled by move? Or just allow them?
            # Actually, adjacent swaps are very common. Let's allow them.
        
        # Identify affected windows
        # Windows affected are those containing i or j.
        # Since i and j are close, the affected region is [min(i,j)-n+1, max(i,j)]
        
        L = len(word)
        min_idx = min(i, j)
        max_idx = max(i, j)
        
        start_k = max(0, min_idx - n + 1)
        end_k = min(max_idx, L - n)
        
        changes = []
        
        # 1. Decrement old windows
        for k in range(start_k, end_k + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                changes.append((pid, -1))
                if counts[pid] < 0:
                    for cid, delta in changes:
                        counts[cid] -= delta
                    return False
        
        # 2. Apply swap in memory temporarily to check new windows?
        # Or calculate new windows directly.
        # It's safer to swap, check, then unswap if invalid.
        word[i], word[j] = word[j], word[i]
        
        # 3. Increment new windows
        for k in range(start_k, end_k + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                changes.append((pid, 1))
        
        # Check validity
        valid = True
        for pid, _ in changes:
            if counts[pid] == 0:
                valid = False
                break
        
        if valid:
            return True
        else:
            # Revert swap
            word[i], word[j] = word[j], word[i]
            # Revert counts
            for cid, delta in changes:
                counts[cid] -= delta
            return False

    def try_move(i, j):
        """Move word[i] to position j (shifting others). Returns True if valid and applied."""
        if i == j:
            return False
        
        # Determine direction
        if i < j:
            # Move right
            # Remove i, insert at j
            # Affected region: i to j+n
            pass
        else:
            # Move left
            pass
            
        # This is complex to verify quickly without doing the move.
        # Given time, I'll stick to Deletions and Swaps for now.
        # Moves are risky and slow to verify.
        return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Deletions
        # Scan backwards
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            changes = is_valid_deletion(i)
            if changes is not None:
                word.pop(i)
                improved = True
                # Continue scanning? Or restart?
                # Restarting ensures we don't miss new opportunities created by the deletion.
                # But restarting is expensive.
                # Let's continue scanning.
                continue
        
        if not improved and time.time() < deadline:
            # Phase 2: Swaps
            # Try swapping characters at distance 1, 2, 3
            for dist in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(len(word) - dist):
                    if time.time() > deadline:
                        break
                    j = i + dist
                    if try_swap(i, j):
                        improved = True
                        # Restart swap phase?
                        # Just continue.
                        break
                if improved:
                    break

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
