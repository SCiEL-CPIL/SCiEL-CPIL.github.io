"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter, defaultdict

def _build_classic_superperm(n):
    """Build the classic recursive superpermutation of 1..n."""
    if n == 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        # Find all (m-1)-element windows in word with no repeats, in order of first appearance
        seen = set()
        order = []
        for i in range(len(word) - m + 2):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                order.append(w)
        new = []
        for p in order:
            block = list(p) + [m] + list(p)
            # Max overlap
            max_ov = min(len(new), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    ov = o
                    break
            new += block[ov:]
        word = new
    return word

def _validate(word, n):
    """Check that all n! permutations of 1..n appear as windows."""
    needed = set(itertools.permutations(range(1, n + 1)))
    found = set()
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        found.add(w)
    return found == needed

def _perm_to_id(perm, n):
    """Convert a permutation tuple to a unique integer ID using factorial number system."""
    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    
    # Map symbols to 0-indexed
    idx = {v: i for i, v in enumerate(range(1, n + 1))}
    arr = [idx[v] for v in perm]
    
    # Compute Lehmuhler code
    code = 0
    used = [False] * n
    for i in range(n):
        count = 0
        for j in range(arr[i]):
            if not used[j]:
                count += 1
        code += count * fact[n - 1 - i]
        used[arr[i]] = True
    return code

def _id_to_perm(pid, n):
    """Convert an integer ID back to a permutation tuple."""
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    
    avail = list(range(n))
    perm = [0] * n
    for i in range(n):
        f = fact[n - 1 - i]
        idx = pid // f
        pid %= f
        perm[i] = avail[idx]
        del avail[idx]
    
    return tuple(v + 1 for v in perm)

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic construction
    word = _build_classic_superperm(n)
    
    if n <= 4:
        return "".join(map(str, word))
    
    # Precompute all permutations and their IDs
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_id_map = {}
    for p in all_perms:
        perm_id_map[p] = _perm_to_id(p, n)
    
    num_perms = len(all_perms)
    
    # Initialize perm counts using IDs
    perm_counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        pid = perm_id_map.get(w, -1)
        if pid >= 0:
            perm_counts[pid] += 1
    
    best_word = list(word)
    best_len = len(word)
    
    # Greedy deletion with multiple passes
    for pass_num in range(20):
        if time.time() - start_time > deadline - 3:
            break
        
        # Different scan order each pass
        positions = list(range(len(word)))
        if pass_num % 2 == 1:
            positions.reverse()
        elif pass_num % 4 == 2:
            positions.sort(key=lambda x: (x * 7919) % max(1, len(word)))
        elif pass_num % 4 == 3:
            positions.sort(key=lambda x: (x * 104729) % max(1, len(word)))
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 3:
            changed = False
            for i in positions:
                if i >= len(word):
                    continue
                if time.time() - start_time > deadline - 3:
                    break
                
                # Check if position i can be safely deleted
                s_min = max(0, i - n + 1)
                s_max = min(i, len(word) - n)
                
                safe = True
                for s in range(s_min, s_max + 1):
                    w = tuple(word[s:s + n])
                    pid = perm_id_map.get(w, -1)
                    if pid >= 0 and perm_counts[pid] == 1:
                        safe = False
                        break
                
                if safe:
                    # Save old windows before deletion
                    old_pids = []
                    for s in range(max(0, i - n + 1), min(len(word) - n + 1, i + 1)):
                        w = tuple(word[s:s + n])
                        pid = perm_id_map.get(w, -1)
                        if pid >= 0:
                            old_pids.append(pid)
                    
                    word.pop(i)
                    changed = True
                    
                    # Remove old window counts
                    for pid in old_pids:
                        perm_counts[pid] -= 1
                    
                    # Add new window counts
                    new_lo = max(0, i - n + 1)
                    new_hi = min(len(word) - n + 1, i + 1)
                    for s in range(new_lo, new_hi):
                        w = tuple(word[s:s + n])
                        pid = perm_id_map.get(w, -1)
                        if pid >= 0:
                            perm_counts[pid] += 1
                    
                    if len(word) < best_len:
                        best_word = list(word)
                        best_len = len(word)
                    
                    # Rebuild positions if length changed
                    if len(word) != len(positions):
                        positions = list(range(len(word)))
                        if pass_num % 2 == 1:
                            positions.reverse()
                        elif pass_num % 4 == 2:
                            positions.sort(key=lambda x: (x * 7919) % max(1, len(word)))
                        elif pass_num % 4 == 3:
                            positions.sort(key=lambda x: (x * 104729) % max(1, len(word)))
        
        # Verify best_word is still valid
        if not _validate(best_word, n):
            best_word = list(word)
    
    if len(best_word) < len(word):
        word = best_word
    
    # Additional local search: try swapping adjacent pairs
    if time.time() - start_time < deadline - 5:
        for _ in range(50):
            if time.time() - start_time > deadline - 3:
                break
            
            # Recompute perm counts for current word
            perm_counts2 = [0] * num_perms
            for i in range(len(word) - n + 1):
                w = tuple(word[i:i + n])
                pid = perm_id_map.get(w, -1)
                if pid >= 0:
                    perm_counts2[pid] += 1
            
            improved = False
            for i in range(len(word) - 1):
                if time.time() - start_time > deadline - 3:
                    break
                
                # Try swapping word[i] and word[i+1]
                word[i], word[i+1] = word[i+1], word[i]
                
                # Check validity quickly
                valid = True
                for s in range(max(0, i - n + 1), min(len(word) - n + 1, i + 2)):
                    w = tuple(word[s:s + n])
                    pid = perm_id_map.get(w, -1)
                    if pid >= 0 and perm_counts2[pid] == 0:
                        # This swap would create a window that wasn't there before
                        # Need to check if it's a new permutation
                        pass
                
                # Simpler: just check if the swap maintains all permutations
                # by verifying no permutation is lost
                # This is expensive, so only do it if the swap seems promising
                word[i], word[i+1] = word[i+1], word[i]  # revert
                # For now, skip swap optimization due to complexity
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
