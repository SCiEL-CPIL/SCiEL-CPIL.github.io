"""Optimized program for minimum maximal no-isosceles set."""
import time
import random
import numpy as np

N = 16
NUM = N * N

# Precompute coordinates
coords = [(i // N, i % N) for i in range(NUM)]

# Precompute squared distances
dist = np.zeros((NUM, NUM), dtype=np.int32)
for i in range(NUM):
    x1, y1 = coords[i]
    for j in range(i + 1, NUM):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i, j] = d
        dist[j, i] = d

# Precompute block_b[p][a] = bitset of b such that triangle(p,a,b) is isosceles
block_b = np.zeros((NUM, NUM), dtype=object)
for p in range(NUM):
    for a in range(NUM):
        if a == p:
            block_b[p, a] = 0
            continue
        bits = 0
        for b in range(NUM):
            if b == p or b == a:
                continue
            d1 = dist[p, a]
            d2 = dist[p, b]
            d3 = dist[a, b]
            if d1 == d2 or d1 == d3 or d2 == d3:
                bits |= (1 << b)
        block_b[p, a] = bits

def can_add(p, S_bits):
    """Check if point p can be added to set S_bits."""
    s = S_bits
    while s:
        low = s & (-s)
        a = low.bit_length() - 1
        if (block_b[p, a] & S_bits) != 0:
            return False
        s ^= low
    return True

def is_maximal(S_bits):
    """Check if set S_bits is maximal."""
    for p in range(NUM):
        if (S_bits >> p) & 1:
            continue
        if can_add(p, S_bits):
            return False
    return True

def make_maximal(S_bits):
    """Add points until maximal."""
    changed = True
    while changed:
        changed = False
        for p in range(NUM):
            if (S_bits >> p) & 1:
                continue
            if can_add(p, S_bits):
                S_bits |= (1 << p)
                changed = True
    return S_bits

def count_bits(x):
    return bin(x).count('1')

def local_search(S_bits, time_limit):
    """Perform local search to minimize the set while keeping it maximal."""
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        s_list = [i for i in range(NUM) if (S_bits >> i) & 1]
        random.shuffle(s_list)
        
        for p in s_list:
            if time.time() > time_limit:
                return S_bits
            
            trial = S_bits & ~(1 << p)
            if is_maximal(trial):
                S_bits = trial
                improved = True
                break
    
    return S_bits

def construct():
    start_time = time.time()
    time_limit = 325.0  # Safety margin
    
    best_size = NUM
    best_bits = 0
    
    random.seed(42)
    
    # Try multiple random restarts with different strategies
    while time.time() - start_time < time_limit:
        strategy = random.random()
        
        if strategy < 0.3:
            # Random greedy
            order = list(range(NUM))
            random.shuffle(order)
        elif strategy < 0.6:
            # Greedy with random tie-breaking, prefer points with fewer conflicts
            # Compute conflict count for each point
            conflict_counts = []
            for p in range(NUM):
                count = 0
                for a in range(NUM):
                    if a == p:
                        continue
                    count += bin(block_b[p, a]).count('1')
                conflict_counts.append((count, p))
            conflict_counts.sort()
            # Take points with fewer conflicts first, with some randomness
            order = [p for _, p in conflict_counts]
            # Add some randomness
            for i in range(len(order) - 1, 0, -1):
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        else:
            # Reverse order (start from corners, etc.)
            order = list(range(NUM))
            random.shuffle(order)
        
        S_bits = 0
        for p in order:
            if can_add(p, S_bits):
                S_bits |= (1 << p)
        
        S_bits = make_maximal(S_bits)
        
        # Local search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            S_bits = local_search(S_bits, start_time + remaining_time - 1)
        
        sz = count_bits(S_bits)
        if sz < best_size:
            best_size = sz
            best_bits = S_bits
    
    # Convert to coordinates
    result = []
    for i in range(NUM):
        if (best_bits >> i) & 1:
            result.append(coords[i])
    
    return result
