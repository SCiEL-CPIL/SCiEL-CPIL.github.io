import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Step 1: Select moduli
    # We hypothesize that all even numbers form a valid set of moduli
    # and that we can find valid residues for them.
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # If we have time, we could try to add odd numbers if they don't conflict,
    # but since evens include powers of 2, no odd number can be added.
    # So we stick with evens.
    
    # Step 2: Assign residues using backtracking
    # Order moduli by ascending value to constrain early?
    # Or maybe descending?
    # Let's try ascending.
    n = len(moduli)
    assignment = [None] * n
    
    # Precompute gcds for speed?
    # gcd_matrix[i][j]
    # n=64, 64x64 matrix is small.
    gcds = [[gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    def is_valid(i, r):
        for j in range(n):
            if assignment[j] is not None:
                g = gcds[i][j]
                if r % g == assignment[j] % g:
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
        
        # Heuristic: Try to find a valid residue for moduli[idx]
        # We can try all residues 0..m-1
        m = moduli[idx]
        
        # Optimization: If we fail, we might want to reorder, but for simplicity,
        # let's just try all residues.
        for r in range(m):
            if is_valid(idx, r):
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = None
        
        return False

    # To improve performance, we can use a smarter ordering or randomized restarts
    # if the first try is slow.
    # But let's try the simple version first.
    
    # If the simple backtracking is too slow, we need a better heuristic.
    # Let's add a timeout check inside the loop or use a limit on depth?
    # The recursion depth is 64, which is fine.
    # The branching factor is up to 128.
    # Worst case is huge, but constraints are tight.
    
    # Let's try a randomized greedy first to get a quick solution, then improve?
    # Or just run the backtracking.
    
    # To be safe against worst-case backtracking explosion,
    # we can limit the search or use a different strategy.
    
    # Strategy: 
    # 1. Try backtracking.
    # 2. If it takes too long, fall back to a greedy solution with a smaller set.
    
    # Let's wrap the backtracking in a try/except or check time.
    # Python recursion limit is usually 1000, so 64 is fine.
    
    success = backtrack(0)
    
    if success:
        result = [[assignment[i], moduli[i]] for i in range(n)]
        return result
    else:
        # Fallback: Greedy with a smaller set
        # Try to build a set greedily
        result = []
        used_moduli = set()
        # Sort moduli by "ease" (e.g., prime factors count?)
        # Just iterate
        for m in moduli:
            found = False
            for r in range(m):
                conflict = False
                for r2, m2 in result:
                    if r % gcd(m, m2) == r2 % gcd(m, m2):
                        conflict = True
                        break
                if not conflict:
                    result.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
            if not found:
                continue
        
        return result
