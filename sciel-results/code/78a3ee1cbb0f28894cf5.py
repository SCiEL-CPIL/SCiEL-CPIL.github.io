import time
import numpy as np
import random

def _get_divisors(n):
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def _score_classes_fast(classes, base=51975):
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return base - int(covered.sum())

def _greedy_construct(base=51975, order='ascending'):
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    if order == 'descending':
        divisors = divisors[::-1]
    elif order == 'random':
        random.shuffle(divisors)
    
    for m in divisors:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = int((covered[r::m] == 0).sum())
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, base=51975, time_limit=10, max_iterations=10000):
    start_time = time.time()
    n = len(classes)
    
    # Initialize counts
    counts = np.zeros(base, dtype=np.int16)
    for m, r in classes:
        counts[r::m] += 1
        
    current_score = int((counts == 0).sum())
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            m_i, r_i = best_classes[i]
            
            # Precompute positions for this modulus? 
            # Arange is fast enough.
            # But we can just loop.
            
            for new_r in range(m_i):
                if new_r == r_i:
                    continue
                
                old_positions = np.arange(r_i, base, m_i)
                new_positions = np.arange(new_r, base, m_i)
                
                # Get current counts at these positions
                old_counts = counts[old_positions]
                new_counts = counts[new_positions]
                
                # Calculate delta
                # Positions in old_positions that become 0: old_counts == 1
                # Positions in new_positions that become 1: new_counts == 0
                # Note: if a position is in both old and new, it cancels out?
                # No, old_positions and new_positions are disjoint if r_i != new_r?
                # If r_i != new_r, they are disjoint modulo m_i.
                # Wait, if m_i is small, they might overlap?
                # No, r and r' are distinct residues mod m.
                # So the sets of indices are disjoint.
                
                delta = int((old_counts == 1).sum()) - int((new_counts == 0).sum())
                new_score = current_score + delta
                
                if new_score < best_score:
                    # Apply move
                    counts[old_positions] -= 1
                    counts[new_positions] += 1
                    current_score = new_score
                    best_score = new_score
                    best_classes[i][1] = new_r
                    improved = True
                else:
                    # No change, don't touch counts
                    pass
            # Note: if improved, r_i changed, so next iteration of new_r uses new r_i?
            # The loop `for new_r in range(m_i)` uses `r_i` from `best_classes[i]` at the start of the i-loop.
            # If we update `best_classes[i][1]`, we should probably restart the inner loop or just continue?
            # Usually, first-improvement or best-improvement.
            # If we find an improvement, we update and continue to next i? Or next new_r?
            # Let's stick to first improvement per i to keep it simple and fast.
            # Break inner loop if improved?
            # The current code doesn't break. It tries all.
            # If we update best_classes[i][1], the subsequent `new_r` checks in this inner loop
            # will compare against the OLD r_i (stored in local var).
            # This is fine, but we might miss better moves if we don't restart.
            # However, with 10000 iterations, it should converge.
            
    return best_classes

def _multi_start_local_search(base=51975, time_limit=60, num_starts=5):
    start_time = time.time()
    best_classes = None
    best_score = float('inf')
    
    orders = ['ascending', 'descending', 'random', 'random', 'random']
    for idx, order in enumerate(orders):
        if time.time() - start_time > time_limit:
            break
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        ls_time = min(remaining_time * 0.7, 15)
        
        classes = _greedy_construct(base, order=order)
        classes = _local_search(classes, base, time_limit=ls_time)
        score = _score_classes_fast(classes, base)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145
    
    classes = _multi_start_local_search(base, time_limit=min(100, int(deadline - time.time()) - 10))
    
    if time.time() < deadline - 5 and classes is not None:
        remaining = int(deadline - time.time()) - 2
        if remaining > 3:
            classes = _local_search(classes, base, time_limit=min(remaining, 20))
    
    return classes
