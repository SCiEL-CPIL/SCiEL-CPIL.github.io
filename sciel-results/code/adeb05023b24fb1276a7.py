"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
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
    
    def try_remove(improved, counts, pos, n):
        """Try to remove position pos from improved. Returns True if removed."""
        j_min = max(0, pos - n + 1)
        j_max = min(pos, len(improved) - n)
        
        safe = True
        affected = []
        for j in range(j_min, j_max + 1):
            w = tuple(improved[j:j+n])
            affected.append(w)
            if counts[w] <= 1:
                safe = False
                break
        
        if safe:
            for w in affected:
                counts[w] -= 1
            improved.pop(pos)
            return True
        return False
    
    def greedy_remove_pass(improved, counts, n, order="fwd"):
        """Do one greedy removal pass in the given order. Returns count of removals."""
        removals = 0
        if order == "fwd":
            i = 0
            while i < len(improved):
                if try_remove(improved, counts, i, n):
                    removals += 1
                else:
                    i += 1
        elif order == "bwd":
            i = len(improved) - 1
            while i >= 0:
                if try_remove(improved, counts, i, n):
                    removals += 1
                else:
                    i -= 1
        elif order == "random":
            # Collect all positions and try in random order, rebuilding as we go
            # This is more expensive but can find different removals
            positions = list(range(len(improved)))
            random.shuffle(positions)
            for pos in positions:
                if 0 <= pos < len(improved):
                    if try_remove(improved, counts, pos, n):
                        removals += 1
        return removals
    
    improved = list(word)
    counts = build_coverage(improved, n)
    
    # Phase 1: Standard greedy removal passes (fwd, bwd, random)
    while time.time() < deadline:
        changed = False
        for order in ["fwd", "bwd", "random"]:
            if time.time() > deadline:
                break
            r = greedy_remove_pass(improved, counts, n, order)
            if r > 0:
                changed = True
        if not changed:
            break
    
    if time.time() > deadline:
        return "".join(map(str, improved))
    
    # Phase 2: Swap-based local search
    # Try swapping adjacent pairs to enable more removals
    # We'll do multiple rounds of: try all adjacent swaps, keep best improvement
    best_len = len(improved)
    best_word = improved[:]
    best_counts = dict(counts)
    
    def evaluate_swap(improved, counts, i, n):
        """Try swapping positions i and i+1. Returns (new_len, new_improved, new_counts) if better, else None."""
        if i + 1 >= len(improved):
            return None
        
        # Try the swap
        improved[i], improved[i+1] = improved[i+1], improved[i]
        
        # Rebuild coverage (expensive but necessary for correctness)
        new_counts = build_coverage(improved, n)
        
        # Check if all permutations are still covered
        # (We trust that if we only do safe removals, coverage is maintained,
        #  but swaps might break it, so we need to verify)
        # Actually, let's just check if the swap enables any removals
        # First, try greedy removal on the swapped version
        test_improved = improved[:]
        test_counts = new_counts
        test_removals = 0
        for _ in range(3):  # A few passes
            r = greedy_remove_pass(test_improved, test_counts, n, "fwd")
            test_removals += r
            if r == 0:
                break
        
        # Check coverage of the test version
        # Count unique n-windows
        unique_windows = set()
        for k in range(len(test_improved) - n + 1):
            unique_windows.add(tuple(test_improved[k:k+n]))
        
        # We need exactly n! unique windows (or more, but all n! must be present)
        target = n
        for _ in range(n):
            target *= _ if _ > 1 else 1
        # Actually n! for n=7 is 5040
        target = 1
        for k in range(2, n+1):
            target *= k
        
        if len(unique_windows) >= target:
            # Verify all permutations are present
            # This is expensive; for n=7, 5040 perms, we can check
            # But it's O(n! * n) which is fine
            perm_set = set(itertools.permutations(range(1, n+1)))
            missing = perm_set - unique_windows
            if not missing:
                # Valid! Keep this swap if it leads to shorter word
                if len(test_improved) < best_len:
                    improved[i], improved[i+1] = improved[i+1], improved[i]  # undo swap
                    return (len(test_improved), test_improved, test_counts)
        
        # Undo the swap
        improved[i], improved[i+1] = improved[i+1], improved[i]
        return None
    
    # Do swap-based search with time budget
    swap_rounds = 0
    while time.time() < deadline - 10:  # Leave 10s buffer
        swap_rounds += 1
        found_improvement = False
        
        # Try all adjacent swaps
        for i in range(len(improved) - 1):
            if time.time() > deadline - 10:
                break
            
            result = evaluate_swap(improved, counts, i, n)
            if result is not None:
                new_len, new_improved, new_counts = result
                improved = new_improved
                counts = new_counts
                best_len = new_len
                best_word = improved[:]
                best_counts = dict(counts)
                found_improvement = True
                break  # Restart scan after improvement
        
        if not found_improvement:
            # No single swap helped; try a few random swaps
            tried = set()
            for _ in range(min(100, len(improved) * 2)):
                if time.time() > deadline - 10:
                    break
                i = random.randint(0, len(improved) - 2)
                if i in tried:
                    continue
                tried.add(i)
                
                result = evaluate_swap(improved, counts, i, n)
                if result is not None:
                    new_len, new_improved, new_counts = result
                    improved = new_improved
                    counts = new_counts
                    best_len = new_len
                    best_word = improved[:]
                    best_counts = dict(counts)
                    found_improvement = True
                    break
        
        if not found_improvement:
            break
    
    # Final greedy removal pass
    for order in ["fwd", "bwd", "random"]:
        if time.time() > deadline - 5:
            break
        greedy_remove_pass(improved, counts, n, order)
    
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
