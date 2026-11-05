import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Greedy earliest-insertion superpermutation
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
            best_overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    S = list(word)
    L = len(S)
    
    # 2. Initialize counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)
    
    # Helper to get window tuple at index i
    def get_window(i):
        return tuple(S[i:i+n])
        
    def try_remove(i):
        """Try to remove S[i]. Returns True if successful."""
        if i < 0 or i >= len(S):
            return False
            
        start = max(0, i - n + 1)
        end = min(i, len(S) - n)
        
        # Check validity: no window containing i has count 1
        for j in range(start, end + 1):
            p = get_window(j)
            if counts.get(p, 0) == 1:
                return False
        
        # Perform removal
        # 1. Decrement counts for destroyed windows
        for j in range(start, end + 1):
            p = get_window(j)
            c = counts[p] - 1
            if c == 0:
                del counts[p]
            else:
                counts[p] = c
        
        # 2. Increment counts for new straddling windows
        new_start = max(0, i - n + 1)
        new_end = min(i - 1, len(S) - 1 - n)
        
        for k in range(new_start, new_end + 1):
            wc = []
            for idx in range(k, k + n):
                if idx < i:
                    wc.append(S[idx])
                else:
                    wc.append(S[idx + 1])
            p_new = tuple(wc)
            counts[p_new] = counts.get(p_new, 0) + 1
            
        S.pop(i)
        return True

    def perform_swap(i, j):
        """Swap S[i] and S[j] and update counts incrementally."""
        if i == j:
            return
        if i > j:
            i, j = j, i
            
        starts = set()
        for k in range(max(0, i - n + 1), min(i, len(S) - n) + 1):
            starts.add(k)
        for k in range(max(0, j - n + 1), min(j, len(S) - n) + 1):
            starts.add(k)
            
        old_windows = {k: get_window(k) for k in starts}
        
        for k, p in old_windows.items():
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
                
        S[i], S[j] = S[j], S[i]
        
        for k in starts:
            p_new = get_window(k)
            counts[p_new] = counts.get(p_new, 0) + 1

    def get_bottleneck_windows():
        """Return list of (window, count) where count == 1, sampled if too many."""
        bottlenecks = [(p, c) for p, c in counts.items() if c == 1]
        if len(bottlenecks) > 100:
            random.shuffle(bottlenecks)
            bottlenecks = bottlenecks[:100]
        return bottlenecks

    def perturb_targeted():
        """Perturb by swapping elements that affect bottleneck windows."""
        bottlenecks = get_bottleneck_windows()
        if not bottlenecks:
            # No bottlenecks, do random swap
            if len(S) > n:
                i, j = random.sample(range(len(S)), 2)
                perform_swap(i, j)
            return
        
        # Pick a random bottleneck window
        target_perm, _ = random.choice(bottlenecks)
        
        # Find all positions where this window occurs
        positions = []
        for i in range(len(S) - n + 1):
            if get_window(i) == target_perm:
                positions.append(i)
        
        if not positions:
            if len(S) > n:
                i, j = random.sample(range(len(S)), 2)
                perform_swap(i, j)
            return
        
        # Pick a random occurrence
        pos = random.choice(positions)
        
        # Try swapping one element in this window with a random element outside
        window_positions = list(range(pos, pos + n))
        
        # Find candidates outside the window
        outside = [i for i in range(len(S)) if i not in set(window_positions)]
        
        if not outside:
            return
        
        # Try a few random swaps
        for _ in range(5):
            i = random.choice(window_positions)
            j = random.choice(outside)
            # Save state
            old_S = S[:]
            old_counts = counts.copy()
            
            perform_swap(i, j)
            
            # Check if this unlocked any removals
            # Quick check: try removing a few random positions
            unlocked = False
            test_positions = random.sample(range(len(S)), min(10, len(S)))
            for tp in test_positions:
                if try_remove(tp):
                    unlocked = True
                    break
            
            if not unlocked:
                # Revert
                S[:] = old_S
                counts.clear()
                counts.update(old_counts)
        
    def multi_remove_pass():
        """Aggressive removal pass: try removing elements from right to left, 
        and if stuck, try swapping to unlock."""
        removed_any = False
        i = len(S) - 1
        while i >= 0 and time.time() < deadline:
            if try_remove(i):
                removed_any = True
            else:
                i -= 1
        return removed_any

    # ILS Loop
    stuck_rounds = 0
    max_stuck = 30
    
    while time.time() < deadline:
        # Multi-pass removal
        removed_any = multi_remove_pass()
        
        if removed_any:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            
            # Perturb: use targeted perturbation
            if len(S) > n:
                if random.random() < 0.7:
                    perturb_targeted()
                else:
                    # Random swap
                    i, j = random.sample(range(len(S)), 2)
                    perform_swap(i, j)
                
    # Final Validation
    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        expected = 1
        for k in range(1, n + 1):
            expected *= k
        return len(seen) == expected

    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
