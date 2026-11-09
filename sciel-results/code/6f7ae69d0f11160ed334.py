import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    """Check if all pair sums (including diagonals) are unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon_4_fast(arr):
    """Check if arr contains any 4-element Sidon subset (faster with early exit)."""
    n = len(arr)
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    a, b, c, d = arr[i], arr[j], arr[k], arr[l]
                    # Check all 10 pair sums (including diagonals)
                    sums = {
                        a + a, a + b, a + c, a + d,
                        b + b, b + c, b + d,
                        c + c, c + d,
                        d + d
                    }
                    if len(sums) == 10:
                        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    random.seed(123)
    
    best_set = list(range(12))
    best_score = 4
    
    # Strategy 1: Try structured dense sets that force collisions
    # These have many repeated differences, creating sum collisions
    
    # Arithmetic progressions with small step
    for step in [1, 2, 3]:
        for offset in range(0, 128 - 11 * step + 1, 1):
            if time.time() > deadline:
                return best_set
            cand = [offset + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
                if not _has_sidon_4_fast(cand):
                    return cand
    
    # Strategy 2: Clustered sets - group numbers in tight clusters
    # If we have many numbers close together, their sums collide
    
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Create clusters of 3-4 numbers
        num_clusters = random.choice([3, 4, 5])
        cluster_sizes = []
        remaining = 12
        for i in range(num_clusters):
            if i == num_clusters - 1:
                cluster_sizes.append(remaining)
            else:
                size = random.randint(2, min(4, remaining))
                cluster_sizes.append(size)
                remaining -= size
        
        cand = []
        used = set()
        for size in cluster_sizes:
            center = random.randint(0, 128)
            for _ in range(size):
                val = center + random.randint(-2, 2)
                if 0 <= val <= 128 and val not in used:
                    cand.append(val)
                    used.add(val)
        
        if len(cand) < 12:
            continue
        
        # Fill remaining with random small values
        while len(cand) < 12:
            val = random.randint(0, 30)
            if val not in used:
                cand.append(val)
                used.add(val)
        
        cand = cand[:12]
        cand.sort()
        
        if not _has_sidon_4_fast(cand):
            return cand
    
    # Strategy 3: Local search - start from dense base and perturb
    base = list(range(12))
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Randomly modify 1-2 elements to create more collisions
        trial = base[:]
        num_mods = random.randint(1, 2)
        
        for _ in range(num_mods):
            idx = random.randint(0, 11)
            old_val = trial[idx]
            
            # Try values that are close to existing elements to force collisions
            if random.random() < 0.7:
                # Pick a value near an existing element
                target_idx = random.randint(0, 11)
                if target_idx == idx:
                    target_idx = (target_idx + 1) % 12
                target_val = trial[target_idx]
                new_val = target_val + random.randint(-3, 3)
            else:
                new_val = random.randint(0, 40)
            
            if 0 <= new_val <= 128 and new_val not in [trial[j] for j in range(12) if j != idx]:
                trial[idx] = new_val
        
        trial.sort()
        
        if not _has_sidon_4_fast(trial):
            return trial
        
        # Occasionally accept changes that don't immediately give 3
        # but might lead to better local optima
        if random.random() < 0.1:
            base = trial[:]
    
    return best_set
