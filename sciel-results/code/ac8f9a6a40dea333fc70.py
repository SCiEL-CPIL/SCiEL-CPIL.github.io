import itertools
import math
import time
import random
import numpy as np

def _score_set_fft(U):
    """Compute score using FFT-based convolution for sumset/difference set sizes."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Sumset: |U+U|
    # Create indicator array
    n = max_u + 1
    f = np.zeros(n, dtype=np.float64)
    for x in U:
        f[x] = 1.0
    
    # Convolve to get sumset
    conv = np.convolve(f, f, mode='full')
    size_s = int(np.sum(conv > 0.5))  # Count positions where sum exists
    
    # Difference set: |U-U|
    # We need to count distinct values of a-b for a,b in U
    # This ranges from -max_u to max_u
    # Use convolution trick: flip one array
    f_flip = f[::-1]
    diff_conv = np.convolve(f, f_flip, mode='full')
    # The difference a-b appears at position (a-b) + (n-1) in the convolution
    size_d = int(np.sum(diff_conv > 0.5))
    
    if size_s == 0:
        return 0.0
        
    ratio = size_d / size_s
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Score function with fallback to brute force for very small sets."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For very small sets, use brute force
    if len(U) <= 10 and max_u <= 50:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        size_s = len(sums)
        size_d = len(diffs)
        if size_s == 0:
            return 0.0
        ratio = size_d / size_s
        if ratio <= 1.0:
            return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    return _score_set_fft(U)

def _golomb_ruler(n, max_val=200):
    """Generate a Golomb-like ruler with n marks using greedy algorithm."""
    if n <= 1:
        return [0]
    
    marks = [0]
    for i in range(1, n):
        # Try to place next mark at smallest valid position
        placed = False
        for pos in range(1, max_val + 1):
            # Check if pos creates duplicate differences
            valid = True
            for j in range(len(marks)):
                diff = pos - marks[j]
                # Check if this difference already exists
                for k in range(len(marks)):
                    if k != j:
                        if abs(marks[j] - marks[k]) == diff and j != k:
                            # This difference already exists between other pairs
                            # Actually we need to check all pairs
                            pass
            # Better: check all existing differences
            existing_diffs = set()
            for a in range(len(marks)):
                for b in range(a + 1, len(marks)):
                    existing_diffs.add(abs(marks[a] - marks[b]))
            
            # Check new differences
            new_diffs = set()
            for j in range(len(marks)):
                new_diffs.add(abs(pos - marks[j]))
            
            if not (new_diffs & existing_diffs):
                marks.append(pos)
                placed = True
                break
        
        if not placed:
            # Couldn't place, just add next integer
            marks.append(marks[-1] + 1)
    
    return marks

def _local_search(U, max_iterations=5000, time_limit=50.0):
    """Local search to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    best_set = U[:]
    best_score = current_score
    
    n = len(U)
    max_u = max(U) if U else 0
    
    for iteration in range(max_iterations):
        if time.time() > start_time + time_limit:
            break
        
        # Random perturbation: add, remove, or modify an element
        op = random.random()
        
        if op < 0.3 and len(U) > 2:
            # Remove a random element (keep 0)
            removable = [x for x in U if x > 0]
            if not removable:
                continue
            idx = random.choice(removable)
            new_U = [x for x in U if x != idx]
            
        elif op < 0.6:
            # Add a random element
            max_range = max_u + random.randint(1, 20)
            new_elem = random.randint(1, max_range)
            if new_elem not in U:
                new_U = U + [new_elem]
            else:
                continue
                
        elif op < 0.8:
            # Modify a random element
            modifiable = [x for x in U if x > 0]
            if not modifiable:
                continue
            idx = random.choice(modifiable)
            new_val = max(1, idx + random.randint(-5, 5))
            if new_val not in U:
                new_U = [new_val if x == idx else x for x in U]
            else:
                continue
                
        else:
            # Swap two elements
            if len(U) < 3:
                continue
            modifiable = [x for x in U if x > 0]
            if len(modifiable) < 2:
                continue
            i, j = random.sample(range(len(U)), 2)
            if U[i] == 0 or U[j] == 0:
                continue
            new_U = U[:]
            new_U[i], new_U[j] = new_U[j], new_U[i]
            # Actually swapping doesn't change the set, so let's do something else
            # Modify one element
            new_U = [x for x in U if x != U[i]]
            new_U.append(max(1, U[i] + random.randint(-3, 3)))
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            continue
            
        new_score = _score_set(new_U)
        if new_score > best_score:
            best_score = new_score
            best_set = new_U[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = 1.0
    
    random.seed(42)
    
    # Heuristic 1: Geometric Progressions
    for r in range(2, 10):
        for n in range(3, 8):
            U = [r**i for i in range(n)]
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if time.time() > deadline:
                return best_set

    # Heuristic 2: Golomb rulers
    for n in range(3, 15):
        for max_val in [50, 100, 200, 500]:
            if time.time() > deadline:
                return best_set
            U = _golomb_ruler(n, max_val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]

    # Heuristic 3: Mian-Chowla sequence
    for n in range(3, 12):
        if time.time() > deadline:
            return best_set
        # Generate Mian-Chowla
        mc = [0, 1]
        for i in range(2, n):
            # Find smallest a such that all pairwise sums are distinct
            a = mc[-1] + 1
            while True:
                # Check if adding a creates duplicate sums
                sums = set()
                valid = True
                for j in range(len(mc) + 1):
                    for k in range(j + 1, len(mc) + 1):
                        x = mc[j] if j < len(mc) else a
                        y = mc[k] if k < len(mc) else a
                        s = x + y
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    mc.append(a)
                    break
                a += 1
        
        U = mc[:n]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]

    # Heuristic 4: Random sets with FFT scoring
    for n in range(3, 30):
        for max_val in [n * 2, n * 5, n * 10, n * 20]:
            for _ in range(200):
                if time.time() > deadline:
                    return best_set
                # Generate n-1 random numbers from 1..max_val
                candidates = list(range(1, max_val + 1))
                if len(candidates) < n - 1:
                    continue
                sample = random.sample(candidates, n - 1)
                U = sorted([0] + sample)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]

    # Heuristic 5: Local search from best set
    if time.time() < deadline - 60:
        refined_set, refined_score = _local_search(best_set, max_iterations=20000, time_limit=50.0)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    # Final polish: try to improve with more iterations if time permits
    if time.time() < deadline - 30:
        refined_set, refined_score = _local_search(best_set, max_iterations=10000, time_limit=25.0)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    return best_set
