"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    deadline = time.time() + 140  # Safety margin before 150s limit

    # 1. Generate standard superpermutation of length 1! + ... + n!
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
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 6:
        return "".join(map(str, word))

    # 2. Setup for optimized local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts of each permutation in the current word
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Fast deletion pass with direct count updates
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        current_L = len(word)
        for i in range(current_L - 1, -1, -1):
            if time.time() > deadline:
                break
                
            start_k_old = max(0, i - n + 1)
            end_k_old = min(i, current_L - n)
            start_k_new = max(0, i - n)
            end_k_new = min(i - 1, current_L - 1 - n)
            
            old_pid_counts = {}
            for k in range(start_k_old, end_k_old + 1):
                p = tuple(word[k:k+n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_pid_counts[pid] = old_pid_counts.get(pid, 0) + 1
            
            new_pid_counts = {}
            for k_new in range(start_k_new, end_k_new + 1):
                w1 = word[k_new:i]
                w2 = word[i+1:k_new+n+1]
                p = tuple(w1 + w2)
                pid = perm_to_id.get(p)
                if pid is not None:
                    new_pid_counts[pid] = new_pid_counts.get(pid, 0) + 1
            
            touched = set(old_pid_counts.keys()) | set(new_pid_counts.keys())
            
            valid = True
            for pid in touched:
                dec = old_pid_counts.get(pid, 0)
                inc = new_pid_counts.get(pid, 0)
                if counts[pid] - dec + inc < 1:
                    valid = False
                    break
            
            if valid:
                word.pop(i)
                for pid, dec in old_pid_counts.items():
                    counts[pid] -= dec
                for pid, inc in new_pid_counts.items():
                    counts[pid] += inc
                improved = True
                current_L -= 1

    # Swap-based local search with broader neighborhood
    if time.time() < deadline - 30:
        swap_improved = True
        while swap_improved and time.time() < deadline - 10:
            swap_improved = False
            current_L = len(word)
            for i in range(current_L - 1):
                if time.time() > deadline - 10:
                    break
                
                word[i], word[i+1] = word[i+1], word[i]
                
                # Try deleting positions near the swap
                found_deletion = False
                for j in range(max(0, i-2), min(i+3, current_L-1)):
                    start_k_old = max(0, j - n + 1)
                    end_k_old = min(j, current_L - n)
                    start_k_new = max(0, j - n)
                    end_k_new = min(j - 1, current_L - 1 - n)
                    
                    old_pid_counts = {}
                    for k in range(start_k_old, end_k_old + 1):
                        p = tuple(word[k:k+n])
                        pid = perm_to_id.get(p)
                        if pid is not None:
                            old_pid_counts[pid] = old_pid_counts.get(pid, 0) + 1
                    
                    new_pid_counts = {}
                    for k_new in range(start_k_new, end_k_new + 1):
                        w1 = word[k_new:j]
                        w2 = word[j+1:k_new+n+1]
                        p = tuple(w1 + w2)
                        pid = perm_to_id.get(p)
                        if pid is not None:
                            new_pid_counts[pid] = new_pid_counts.get(pid, 0) + 1
                    
                    touched = set(old_pid_counts.keys()) | set(new_pid_counts.keys())
                    can_delete = True
                    for pid in touched:
                        dec = old_pid_counts.get(pid, 0)
                        inc = new_pid_counts.get(pid, 0)
                        if counts[pid] - dec + inc < 1:
                            can_delete = False
                            break
                    
                    if can_delete:
                        word.pop(j)
                        for pid, dec in old_pid_counts.items():
                            counts[pid] -= dec
                        for pid, inc in new_pid_counts.items():
                            counts[pid] += inc
                        swap_improved = True
                        found_deletion = True
                        break
                
                if not found_deletion:
                    word[i], word[i+1] = word[i+1], word[i]

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
