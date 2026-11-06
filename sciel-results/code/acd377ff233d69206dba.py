import time
import random
import numpy as np

BASE = 51975

# Generate divisors
def get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

DIVS = get_divisors(BASE)
# Sort for consistent processing, maybe largest first for greedy?
# We will handle ordering in the algorithm.

# Precompute indices for each (m, r)
# indices[m][r] is a numpy array of indices x such that x % m == r
INDICES = {}
for m in DIVS:
    INDICES[m] = []
    for r in range(m):
        INDICES[m].append(np.arange(r, BASE, m, dtype=np.int32))

def _score(count):
    """Calculate number of uncovered residues."""
    return np.sum(count == 0)

def _greedy_build(order, rng):
    """
    Build a solution greedily using the given order of moduli.
    """
    count = np.zeros(BASE, dtype=np.int32)
    classes = []
    
    for m in order:
        # Find best r for this m
        best_r = 0
        best_gain = -1
        # To speed up, we can sample if m is large, but m is small enough (max 51975)
        # However, iterating all r for large m is expensive if done naively.
        # But we have precomputed indices.
        # Cost: m * (length of index array) operations? No.
        # We just need to check how many 0s are in count[indices[m][r]].
        
        # Optimization: 
        # If m is large, the index array is small.
        # If m is small, the index array is large, but m is small.
        
        for r in range(m):
            idx = INDICES[m][r]
            # Count how many of these indices are currently uncovered (count == 0)
            gain = np.sum(count[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
                
        # Apply
        count[INDICES[m][best_r]] += 1
        classes.append((m, best_r))
        
    return count, classes

def _local_search(count, classes, budget):
    """
    Improve the solution via local search.
    """
    n = len(classes)
    improved = True
    start_time = time.time()
    
    while improved and time.time() < budget:
        improved = False
        # Shuffle order to avoid bias? Or keep fixed?
        # Fixed order is deterministic. Let's stick to it for simplicity, 
        # but random restarts will handle the landscape.
        
        for i in range(n):
            if time.time() > budget:
                break
                
            m, r = classes[i]
            idx_old = INDICES[m][r]
            
            # Temporarily remove
            count[idx_old] -= 1
            
            # Find best new r for this m
            best_r = r
            best_gain = -1
            
            # We want to maximize the number of newly covered residues.
            # This is equivalent to maximizing sum(count[idx] == 0) for the new r.
            # Note: count reflects the state WITHOUT class i.
            
            for r_new in range(m):
                idx_new = INDICES[m][r_new]
                gain = np.sum(count[idx_new] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r_new
                    
            if best_r != r:
                # Apply new r
                count[INDICES[m][best_r]] += 1
                classes[i] = (m, best_r)
                improved = True
            else:
                # Restore old r
                count[idx_old] += 1
                
    return count, classes

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin
    
    best_score = float('inf')
    best_classes = None
    
    # Strategy: Multiple restarts with different greedy orders
    # 1. Largest modulus first
    # 2. Smallest modulus first
    # 3. Random orders
    
    orders = [
        sorted(DIVS, reverse=True),
        sorted(DIVS),
    ]
    
    # Add some random orders
    rng = random.Random(42)
    for _ in range(10):
        o = DIVS[:]
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 5.0:
            break
            
        count, classes = _greedy_build(order, rng)
        
        # Local search
        budget = min(deadline - 2.0, time.time() + 10.0) # Cap local search time per restart?
        # Actually, let's just use the global deadline.
        count, classes = _local_search(count, classes, deadline)
        
        score = _score(count)
        
        if score < best_score:
            best_score = score
            best_classes = list(classes)
            
            # If we found a perfect covering, we can stop early?
            # Unlikely, but possible.
            if best_score == 0:
                break
                
    if best_classes is None:
        # Fallback
        best_classes = [[m, 0] for m in DIVS]
        
    return [[m, r] for m, r in best_classes]
