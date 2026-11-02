"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

_start_time = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill

def _score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return -1.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    max_u = max(U)
    if max_u == 0:
        return -1.0
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):  # since a+b = b+a, use i<=j
            sumset.add(U[i] + U[j])
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    
    # Difference set is symmetric, so |U-U| = 2*|U-U positive| + 1
    # But we just use the raw set size
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return -1.0
    
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score

def _score_set_fast(U):
    """Faster scoring using numpy for larger sets."""
    import numpy as np
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return -1.0
    max_u = int(U[-1])
    if max_u == 0:
        return -1.0
    
    # Sumset: U[i] + U[j] for i <= j
    # Use broadcasting but be careful about memory
    if n <= 200:
        sums = U[:, None] + U[None, :]
        # Only count unique sums
        sumset_size = len(np.unique(sums))
        diffs = U[:, None] - U[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        # Fall back to set-based for large n
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(int(U[i]) + int(U[j]))
        sumset_size = len(sumset)
        
        diffset = set()
        for i in range(n):
            for j in range(n):
                diffset.add(int(U[i]) - int(U[j]))
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 1:
        return -1.0
    
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score

def _generate_structural_candidates():
    """Generate various structured candidate sets."""
    candidates = []
    
    # Family 1: 2^k - 1
    for m in range(3, 25):
        U = [2**k - 1 for k in range(m + 1)]
        candidates.append(set(U))
    
    # Family 2: powers of 2
    for m in range(3, 25):
        U = [2**k for k in range(m + 1)]
        candidates.append(set(U))
    
    # Family 3: 3^k
    for m in range(2, 15):
        U = [3**k for k in range(m + 1)]
        candidates.append(set(U))
    
    # Family 4: mixed - {0} union {a^k}
    for base in [2, 3, 4, 5]:
        for m in range(3, 20):
            U = [base**k for k in range(m + 1)]
            candidates.append(set(U))
    
    # Family 5: {0, 1, 3, 7, 15, ...} with perturbations
    # Family 6: geometric with different bases
    for base in range(2, 10):
        for m in range(3, 12):
            U = [base**k for k in range(m + 1)]
            candidates.append(set(U))
    
    # Family 7: {0} ∪ {k*a + b for ...} arithmetic + geometric hybrid
    # Family 8: Known good - use the seed and variations
    candidates.append({0, 1, 3})
    candidates.append({0, 1, 3, 7, 15})
    candidates.append({0, 1, 3, 7, 15, 31, 63})
    
    return candidates

def _local_search(initial_set, max_iterations=5000):
    """Local search: add, remove, or replace elements to improve score."""
    current = set(initial_set)
    current_score = _score_set(current)
    if current_score < 0:
        return current, -1.0
    
    best_set = set(current)
    best_score = current_score
    
    for iteration in range(max_iterations):
        if time.time() - _start_time > _DEADLINE:
            break
        
        n = len(current)
        max_val = max(current) if current else 0
        
        # Generate neighbors
        neighbors = []
        
        # 1. Add a new element
        # Try adding elements near existing ones or at strategic positions
        add_candidates = []
        for v in current:
            for delta in [1, 2, 3, 5, 8, 13, 21]:
                add_candidates.append(v + delta)
        # Also try some random additions
        for _ in range(10):
            add_candidates.append(random.randint(1, max_val + 20))
        # Try doubling
        for v in current:
            add_candidates.append(2 * v)
        
        # 2. Remove an element (if size > 2)
        if n > 2:
            remove_candidates = list(current - {0})  # keep 0
        
        # 3. Replace an element
        replace_candidates = []
        if n > 2:
            for v in (current - {0}):
                for delta in [-3, -2, -1, 1, 2, 3, 5]:
                    new_v = v + delta
                    if new_v > 0 and new_v not in current:
                        replace_candidates.append((v, new_v))
        
        # Evaluate neighbors (sample if too many)
        all_neighbors = []
        
        # Add neighbors
        for v in add_candidates[:30]:
            if v not in current and v >= 0:
                all_neighbors.append(('add', v))
        
        # Remove neighbors
        if n > 2:
            for v in remove_candidates[:10]:
                all_neighbors.append(('remove', v))
        
        # Replace neighbors
        for (old, new) in replace_candidates[:15]:
            all_neighbors.append(('replace', (old, new)))
        
        # Shuffle and evaluate
        random.shuffle(all_neighbors)
        improved = False
        
        for action, data in all_neighbors:
            if time.time() - _start_time > _DEADLINE:
                break
            
            new_set = set(current)
            if action == 'add':
                new_set.add(data)
            elif action == 'remove':
                new_set.discard(data)
                if 0 not in new_set:
                    continue
            elif action == 'replace':
                old, new = data
                new_set.discard(old)
                new_set.add(new)
                if 0 not in new_set:
                    continue
            
            if len(new_set) < 2:
                continue
            
            new_score = _score_set(new_set)
            if new_score > current_score + 1e-12:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_set = set(current)
                    best_score = current_score
                improved = True
                break
        
        if not improved:
            # Try a few random perturbations
            for _ in range(5):
                if time.time() - _start_time > _DEADLINE:
                    break
                new_set = set(current)
                # Random add
                if random.random() < 0.5:
                    v = random.randint(1, max_val + 50)
                    new_set.add(v)
                else:
                    # Random replace
                    if len(new_set) > 2:
                        non_zero = list(new_set - {0})
                        if non_zero:
                            old = random.choice(non_zero)
                            new_set.discard(old)
                            new_set.add(random.randint(1, max_val + 50))
                
                if len(new_set) < 2 or 0 not in new_set:
                    continue
                
                new_score = _score_set(new_set)
                if new_score > current_score + 1e-12:
                    current = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_set = set(current)
                        best_score = current_score
                    break
    
    return best_set, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_set = {0, 1, 3}
    best_score = _score_set(best_set)
    
    # Step 1: Evaluate structural candidates
    candidates = _generate_structural_candidates()
    for cand in candidates:
        if time.time() - _start_time > _DEADLINE:
            break
        if 0 not in cand:
            cand = cand | {0}
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = set(cand)
    
    # Step 2: Local search from best structural candidate
    if time.time() - _start_time < _DEADLINE - 5:
        best_set, best_score = _local_search(best_set, max_iterations=3000)
    
    # Step 3: Try a few random starts with local search
    remaining_time = _DEADLINE - (time.time() - _start_time)
    if remaining_time > 10:
        for attempt in range(5):
            if time.time() - _start_time > _DEADLINE:
                break
            
            # Generate a random structured set
            base = random.choice([2, 3, 4, 5])
            m = random.randint(3, 15)
            U = set()
            for k in range(m + 1):
                U.add(base ** k)
            # Add some random elements
            for _ in range(random.randint(0, 5)):
                U.add(random.randint(1, base ** m + 10))
            U.add(0)
            
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = set(U)
            
            # Local search
            if time.time() - _start_time < _DEADLINE - 5:
                new_set, new_score = _local_search(U, max_iterations=1000)
                if new_score > best_score:
                    best_score = new_score
                    best_set = new_set
    
    # Ensure 0 is in the set and return as sorted list
    result = sorted(best_set | {0})
    return result
# EVOLVE-BLOCK-END
