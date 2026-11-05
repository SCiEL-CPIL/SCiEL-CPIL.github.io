import time
import random

def _creates_4ap(x, A):
    """Check if adding x to set A creates any nonconstant 4-term AP."""
    for d in range(1, 86):
        if x + 3*d <= 256 and (x+d in A) and (x+2*d in A) and (x+3*d in A):
            return True
        if x - d >= 1 and (x-d in A) and (x+d in A) and (x+2*d in A):
            return True
        if x - 2*d >= 1 and (x-2*d in A) and (x-d in A) and (x+d in A):
            return True
        if x - 3*d >= 1 and (x-3*d in A) and (x-2*d in A) and (x-d in A):
            return True
    return False

def _gain_3aps(x, A):
    """Count 3-APs created by adding x to set A."""
    gain = 0
    for d in range(1, 257):
        if x - d >= 1 and x + d <= 256:
            if (x-d in A) and (x+d in A):
                gain += 1
        if x + 2*d <= 256:
            if (x+d in A) and (x+2*d in A):
                gain += 1
        if x - 2*d >= 1:
            if (x-2*d in A) and (x-d in A):
                gain += 1
    return gain

def _count_3aps(A):
    """Count all 3-term APs in set A."""
    count = 0
    for b in A:
        d = 1
        while b - d >= 1 and b + d <= 256:
            if (b - d) in A and (b + d) in A:
                count += 1
            d += 1
    return count

def _local_search(A, current_score, deadline):
    """Improve A via add and swap moves."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements with positive 3-AP gain
        for x in range(1, 257):
            if x in A:
                continue
            if _creates_4ap(x, A):
                continue
            g = _gain_3aps(x, A)
            if g > 0:
                A.add(x)
                current_score += g
                improved = True
                break
        
        if time.time() >= deadline:
            break
        
        # Try swaps
        if not improved and time.time() < deadline:
            for x in list(A):
                if time.time() >= deadline:
                    break
                A_temp = A - {x}
                # Compute loss from removing x
                loss = 0
                for d in range(1, 257):
                    if x - d >= 1 and x + d <= 256:
                        if (x-d in A_temp) and (x+d in A_temp):
                            loss += 1
                    if x + 2*d <= 256:
                        if (x+d in A_temp) and (x+2*d in A_temp):
                            loss += 1
                    if x - 2*d >= 1:
                        if (x-2*d in A_temp) and (x-d in A_temp):
                            loss += 1
                
                for y in range(1, 257):
                    if y in A_temp:
                        continue
                    if _creates_4ap(y, A_temp):
                        continue
                    g = _gain_3aps(y, A_temp)
                    if g - loss > 0:
                        A = A_temp | {y}
                        current_score += g - loss
                        improved = True
                        break
                if improved:
                    break
    
    return A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 78.0
    
    # Phase 1: Greedy construction maximizing 3-AP gain
    A = set()
    remaining = set(range(1, 257))
    
    while remaining and time.time() < deadline:
        best_x = None
        best_gain = -1
        for x in remaining:
            if _creates_4ap(x, A):
                continue
            g = _gain_3aps(x, A)
            if g > best_gain:
                best_gain = g
                best_x = x
        if best_x is None:
            break
        A.add(best_x)
        remaining.remove(best_x)
    
    current_score = _count_3aps(A)
    
    # Phase 2: Local search
    A, current_score = _local_search(A, current_score, deadline)
    
    # Phase 3: Iterated local search with random restarts
    if time.time() < deadline - 5.0:
        best_A = set(A)
        best_score = current_score
        
        while time.time() < deadline - 2.0:
            perturbed = set(best_A)
            k = random.randint(3, min(10, len(perturbed)))
            to_remove = random.sample(list(perturbed), k)
            for x in to_remove:
                perturbed.discard(x)
            
            # Greedy refill
            remaining = set(range(1, 257)) - perturbed
            while remaining and time.time() < deadline - 2.0:
                best_x = None
                best_gain = -1
                for x in remaining:
                    if _creates_4ap(x, perturbed):
                        continue
                    g = _gain_3aps(x, perturbed)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x is None:
                    break
                perturbed.add(best_x)
                remaining.remove(best_x)
            
            # Local search on perturbed
            score = _count_3aps(perturbed)
            perturbed, score = _local_search(perturbed, score, deadline - 2.0)
            
            if score > best_score:
                best_score = score
                best_A = set(perturbed)
    
    A = best_A if 'best_A' in dir() else A
    return sorted(A)
