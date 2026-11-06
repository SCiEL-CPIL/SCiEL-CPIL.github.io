"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    def is_sidon(elems):
        """Check if all pair sums (including diagonal) are unique."""
        n = len(elems)
        seen = set()
        for i in range(n):
            for j in range(i, n):
                s = elems[i] + elems[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    def max_sidon_size(S):
        """Find largest Sidon subset of S (12 elements)."""
        n = len(S)
        for size in range(n, 0, -1):
            for subset in combinations(range(n), size):
                elems = [S[i] for i in subset]
                if is_sidon(elems):
                    return size
        return 1
    
    def count_collisions(S):
        """Count pair-sum collisions as a proxy (faster)."""
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                sums[s] = sums.get(s, 0) + 1
        total = 0
        for cnt in sums.values():
            if cnt > 1:
                total += cnt * (cnt - 1) // 2
        return total
    
    rng = random.Random(42)
    
    # Start with a structured initial guess
    # Try to create many collisions by using arithmetic progressions or similar
    best_set = list(range(12))
    best_ms = max_sidon_size(best_set)
    
    current = list(best_set)
    current_coll = count_collisions(current)
    
    while time.time() < deadline:
        improved = False
        for _ in range(500):
            if time.time() > deadline:
                break
            i = rng.randrange(12)
            old_val = current[i]
            new_val = rng.randrange(129)
            if new_val == old_val or new_val in current:
                continue
            
            current[i] = new_val
            coll = count_collisions(current)
            if coll > current_coll:
                # Verify with actual max Sidon size
                ms = max_sidon_size(current)
                if ms < best_ms:
                    best_ms = ms
                    best_set = list(current)
                current_coll = coll
                improved = True
            else:
                current[i] = old_val
        
        if not improved:
            # Restart from best
            current = list(best_set)
            current_coll = count_collisions(current)
    
    return best_set
# EVOLVE-BLOCK-END
