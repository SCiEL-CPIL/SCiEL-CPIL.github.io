import time
import random
from collections import deque

N = 12
TOTAL_MASKS = 1 << N
ALL_STATES = TOTAL_MASKS - 1

# Precompute bit positions for fast iteration
BIT_POSITIONS = [i for i in range(N)]

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    """
    # Precompute transition tables for bitmask application
    trans_a = [0] * TOTAL_MASKS
    trans_b = [0] * TOTAL_MASKS
    
    for mask in range(1, TOTAL_MASKS):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            res_a |= (1 << a_trans[i])
            res_b |= (1 << b_trans[i])
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b

    start = ALL_STATES
    
    # Check if already singleton (shouldn't happen for N>1 with full set)
    if (start & (start - 1)) == 0:
        return 0

    # BFS
    visited = [-1] * TOTAL_MASKS
    visited[start] = 0
    q = [start]
    head = 0
    
    while head < len(q):
        curr = q[head]
        head += 1
        curr_dist = visited[curr]
        
        # Check if singleton
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit
    
    best_score = -1
    best_a = None
    best_b = None
    
    def update(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = list(a)
            best_b = list(b)
        return score
    
    # Strategy: Large-scale random restart local search
    # Directly optimize both transition functions
    # Start with diverse random initializations and hill climb
    
    random.seed(12345)
    
    # Phase 1: Generate diverse starting points
    # Known good structures to seed the search
    starting_points = []
    
    # 1. Pinsker-like: a is cycle, b fixes one state and cycles rest
    a_cycle = [(i + 1) % N for i in range(N)]
    for fixed in range(N):
        b = [0] * N
        b[fixed] = fixed
        others = [i for i in range(N) if i != fixed]
        for i in range(len(others)):
            b[others[i]] = others[(i + 1) % len(others)]
        starting_points.append((a_cycle, b))
    
    # 2. Random permutations for both a and b
    for _ in range(100):
        a = list(range(N))
        b = list(range(N))
        random.shuffle(a)
        random.shuffle(b)
        starting_points.append((a, b))
    
    # 3. a as various cycles, b random
    for d in range(1, N):
        a_shift = [(i + d) % N for i in range(N)]
        for _ in range(20):
            b = list(range(N))
            random.shuffle(b)
            starting_points.append((a_shift, b))
    
    # 4. Random non-permutations (some states map to same destination)
    for _ in range(200):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        starting_points.append((a, b))
    
    # Evaluate all starting points
    for a, b in starting_points:
        if time.time() > deadline:
            break
        update(a, b)
    
    # Phase 2: Local search from best point
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
        update(best_a, best_b)
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_score = best_score
    
    # Simulated annealing style local search
    temperature = 10.0
    cooling_rate = 0.995
    
    iterations = 0
    while time.time() < deadline and temperature > 0.1:
        iterations += 1
        
        # Choose what to mutate
        if random.random() < 0.5:
            # Mutate b
            b_new = list(current_b)
            num_mut = random.randint(1, 3)
            for _ in range(num_mut):
                idx = random.randint(0, N-1)
                val = random.randint(0, N-1)
                b_new[idx] = val
            score = calculate_threshold(current_a, b_new)
            if score > current_score:
                current_b = b_new
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)
            elif random.random() < min(1.0, (score - current_score) / temperature if score >= current_score else 0.1):
                # Accept worse solutions with some probability
                current_b = b_new
                current_score = score
        else:
            # Mutate a
            a_new = list(current_a)
            num_mut = random.randint(1, 3)
            for _ in range(num_mut):
                idx = random.randint(0, N-1)
                val = random.randint(0, N-1)
                a_new[idx] = val
            score = calculate_threshold(a_new, current_b)
            if score > current_score:
                current_a = a_new
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)
            elif random.random() < min(1.0, (score - current_score) / temperature if score >= current_score else 0.1):
                # Accept worse solutions with some probability
                current_a = a_new
                current_score = score
        
        temperature *= cooling_rate
    
    # Phase 3: Intensive local search around the best solution
    if time.time() < deadline - 5.0:
        # Try single-position mutations on both a and b
        for pos in range(N):
            for val in range(N):
                if time.time() > deadline - 2.0:
                    break
                
                # Try changing a[pos]
                a_new = list(best_a)
                a_new[pos] = val
                score = calculate_threshold(a_new, best_b)
                if score > best_score:
                    best_score = score
                    best_a = a_new
                
                # Try changing b[pos]
                b_new = list(best_b)
                b_new[pos] = val
                score = calculate_threshold(best_a, b_new)
                if score > best_score:
                    best_score = score
                    best_b = b_new
        
        # Try two-position mutations
        for pos1 in range(N):
            for pos2 in range(pos1 + 1, N):
                if time.time() > deadline - 1.0:
                    break
                
                for val1 in range(N):
                    for val2 in range(N):
                        if time.time() > deadline - 0.5:
                            break
                        
                        # Try changing a[pos1] and a[pos2]
                        a_new = list(best_a)
                        a_new[pos1] = val1
                        a_new[pos2] = val2
                        score = calculate_threshold(a_new, best_b)
                        if score > best_score:
                            best_score = score
                            best_a = a_new
                        
                        # Try changing b[pos1] and b[pos2]
                        b_new = list(best_b)
                        b_new[pos1] = val1
                        b_new[pos2] = val2
                        score = calculate_threshold(best_a, b_new)
                        if score > best_score:
                            best_score = score
                            best_b = b_new
        
        # Try mixed mutations: one in a, one in b
        if time.time() < deadline - 0.5:
            for pos_a in range(N):
                for pos_b in range(N):
                    if time.time() > deadline - 0.2:
                        break
                    
                    for val_a in range(N):
                        for val_b in range(N):
                            if time.time() > deadline - 0.1:
                                break
                            
                            a_new = list(best_a)
                            b_new = list(best_b)
                            a_new[pos_a] = val_a
                            b_new[pos_b] = val_b
                            score = calculate_threshold(a_new, b_new)
                            if score > best_score:
                                best_score = score
                                best_a = a_new
                                best_b = b_new
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
