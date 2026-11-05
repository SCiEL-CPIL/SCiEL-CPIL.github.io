"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    def get_pair_sums(s):
        """Get all unordered pair sums including diagonals (i<=j)."""
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        return sums
    
    def count_collisions(s):
        """Count how many pair sums are duplicates (total pairs - unique sums)."""
        sums = get_pair_sums(s)
        return len(sums) - len(set(sums))
    
    def max_sidon_size(s):
        """Find the largest subset of s that is Sidon (all pair sums i<=j unique)."""
        n = len(s)
        # Try subsets from largest to smallest
        for k in range(n, 0, -1):
            for subset in combinations(s, k):
                sums = []
                for i in range(k):
                    for j in range(i, k):
                        sums.append(subset[i] + subset[j])
                if len(sums) == len(set(sums)):
                    return k
        return 1
    
    def max_sidon_size_fast(s):
        """Faster check: only check up to a reasonable size, use early termination."""
        n = len(s)
        # For 12 elements, check k from n down
        for k in range(n, 3, -1):
            found = False
            for subset in combinations(s, k):
                sums = set()
                valid = True
                for i in range(k):
                    for j in range(i, k):
                        sm = subset[i] + subset[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                    if not valid:
                        break
                if valid:
                    found = True
                    break
            if found:
                return k
        return 3  # At least 3 elements can form a Sidon set typically
    
    best_set = None
    best_sidon = 99
    best_collision = -1
    
    # Phase 1: Random initialization
    for trial in range(200):
        if time.time() > deadline:
            break
        s = random.sample(range(129), 12)
        s.sort()
        col = count_collisions(s)
        if best_set is None or col > best_collision:
            best_collision = col
            best_set = s[:]
    
    # Phase 2: Local search - swap elements
    if best_set is not None:
        current = best_set[:]
        current_col = best_collision
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping one element with a random value in [0,128]
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = random.randint(0, 11)
                old_val = current[idx]
                new_val = random.randint(0, 128)
                if new_val == old_val:
                    continue
                if new_val in current:
                    continue
                
                candidate = current[:]
                candidate[idx] = new_val
                candidate.sort()
                
                new_col = count_collisions(candidate)
                if new_col > current_col:
                    current = candidate
                    current_col = new_col
                    improved = True
                    break
                elif new_col == current_col:
                    # Accept with some probability for diversity
                    if random.random() < 0.1:
                        current = candidate
                        current_col = new_col
            
            if not improved and random.random() < 0.05:
                # Occasional restart
                current = random.sample(range(129), 12)
                current.sort()
                current_col = count_collisions(current)
        
        if current_col > best_collision:
            best_collision = current_col
            best_set = current[:]
    
    # Phase 3: Evaluate actual Sidon size and refine
    if best_set is not None:
        # Verify the actual max Sidon subset size
        actual_sidon = max_sidon_size_fast(best_set)
        best_sidon = actual_sidon
        
        # Try to further reduce by targeting specific collisions
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            candidate = best_set[:]
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val == candidate[idx]:
                continue
            if new_val in candidate:
                continue
            
            candidate[idx] = new_val
            candidate.sort()
            
            new_col = count_collisions(candidate)
            if new_col >= best_collision:
                # Check if Sidon size decreased
                new_sidon = max_sidon_size_fast(candidate)
                if new_sidon < best_sidon:
                    best_set = candidate
                    best_sidon = new_sidon
                    best_collision = new_col
                elif new_sidon == best_sidon and new_col > best_collision:
                    best_set = candidate
                    best_collision = new_col
    
    # Final: ensure we return a valid set
    if best_set is None:
        best_set = list(range(12))
    
    # Verify validity
    if len(set(best_set)) != 12 or any(x < 0 or x > 128 for x in best_set):
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
