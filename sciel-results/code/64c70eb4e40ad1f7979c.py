import time
import random
from collections import deque

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Returns the length of the shortest reset word.
    """
    full_set = (1 << N) - 1
    visited = {full_set: 0}
    queue = deque([(full_set, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask.bit_count() == 1:
            return dist
        
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= (1 << trans_a[i])
        
        if new_mask_a not in visited:
            visited[new_mask_a] = dist + 1
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= (1 << trans_b[i])
        
        if new_mask_b not in visited:
            visited[new_mask_b] = dist + 1
            queue.append((new_mask_b, dist + 1))
            
    return -1

def construct():
    """
    Systematically searches parameterized families of synchronizing automata
    to find one with maximum reset threshold.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    def fast_threshold(trans_a, trans_b):
        """Fast BFS with precomputed actions."""
        full_set = (1 << N) - 1
        
        # Precompute transitions for all masks
        actions_a = [0] * (1 << N)
        actions_b = [0] * (1 << N)
        for mask in range(1 << N):
            ma = 0
            mb = 0
            for i in range(N):
                if mask & (1 << i):
                    ma |= (1 << trans_a[i])
                    mb |= (1 << trans_b[i])
            actions_a[mask] = ma
            actions_b[mask] = mb
        
        visited = {full_set: 0}
        queue = deque([(full_set, 0)])
        
        while queue:
            mask, dist = queue.popleft()
            if mask.bit_count() == 1:
                return dist
            
            na = actions_a[mask]
            if na not in visited:
                visited[na] = dist + 1
                queue.append((na, dist + 1))
                
            nb = actions_b[mask]
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))
        
        return -1
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Family 1: Černý/Pawlik automaton
    # a: i -> (i+1) % N
    # b: i -> i-1 if i > 0 else 0
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [i - 1 if i > 0 else 0 for i in range(N)]
    score1 = fast_threshold(a1, b1)
    if score1 > best_score:
        best_score = score1
        best_a = a1[:]
        best_b = b1[:]
    
    # Family 2: Cycle + constant reset
    # a: i -> (i+1) % N
    # b: i -> 0 for all i
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [0] * N
    score2 = fast_threshold(a2, b2)
    if score2 > best_score:
        best_score = score2
        best_a = a2[:]
        best_b = b2[:]
    
    # Family 3: Two-cycle structure with reset
    # a: i -> (i+1) % N
    # b: i -> i//2 if i even, (i+1)//2 if i odd (contracting map)
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [i // 2 if i % 2 == 0 else (i + 1) // 2 for i in range(N)]
    score3 = fast_threshold(a3, b3)
    if score3 > best_score:
        best_score = score3
        best_a = a3[:]
        best_b = b3[:]
    
    # Family 4: Parameterized cycle + partial reset
    # a: i -> (i+1) % N
    # b: maps first k states to 0, rest to i-1
    for k in range(1, N):
        a4 = [(i + 1) % N for i in range(N)]
        b4 = [0 if i < k else i - 1 for i in range(N)]
        score4 = fast_threshold(a4, b4)
        if score4 > best_score:
            best_score = score4
            best_a = a4[:]
            best_b = b4[:]
        if time.time() > deadline:
            break
    
    # Family 5: Reverse cycle + constant
    a5 = [(i - 1) % N for i in range(N)]
    b5 = [0] * N
    score5 = fast_threshold(a5, b5)
    if score5 > best_score:
        best_score = score5
        best_a = a5[:]
        best_b = b5[:]
    
    # Family 6: Non-trivial permutation + reset
    # a: swaps pairs, b: resets to 0
    # Try various permutations
    import itertools
    perms = [
        [(i + 1) % N for i in range(N)],  # cycle
        [(i - 1) % N for i in range(N)],  # reverse cycle
        [(-i) % N for i in range(N)],     # reflection
        [(2 * i) % N for i in range(N)],  # doubling (if N even)
        [(3 * i) % N for i in range(N)],  # tripling
    ]
    for perm in perms:
        if len(set(perm)) != N:  # skip if not a permutation
            continue
        for reset_target in range(N):
            a6 = perm[:]
            b6 = [reset_target] * N
            score6 = fast_threshold(a6, b6)
            if score6 > best_score:
                best_score = score6
                best_a = a6[:]
                best_b = b6[:]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Family 7: Pawlik-type with parameter
    # a: i -> (i+1) % N
    # b: i -> 0 if i == 0, i-1 if 1 <= i <= k, i if i > k
    for k in range(1, N):
        a7 = [(i + 1) % N for i in range(N)]
        b7 = [0 if i == 0 else (i - 1 if i <= k else i) for i in range(N)]
        score7 = fast_threshold(a7, b7)
        if score7 > best_score:
            best_score = score7
            best_a = a7[:]
            best_b = b7[:]
        if time.time() > deadline:
            break
    
    # Family 8: Bidirectional cycle with asymmetric reset
    # a: i -> (i+1) % N
    # b: i -> (i-1) % N if i > 0, 0 if i == 0, but with a "hole"
    for hole_pos in range(N):
        a8 = [(i + 1) % N for i in range(N)]
        b8 = []
        for i in range(N):
            if i == 0:
                b8.append(0)
            elif i == hole_pos:
                b8.append(0)  # Reset from hole position
            else:
                b8.append(i - 1)
        score8 = fast_threshold(a8, b8)
        if score8 > best_score:
            best_score = score8
            best_a = a8[:]
            best_b = b8[:]
        if time.time() > deadline:
            break
    
    # Family 9: Known high-threshold structure
    # Based on literature: cycle with a "backbone" reset
    a9 = [(i + 1) % N for i in range(N)]
    b9 = []
    for i in range(N):
        if i == 0:
            b9.append(0)
        elif i % 2 == 0:
            b9.append(i - 2)
        else:
            b9.append(i - 1)
    score9 = fast_threshold(a9, b9)
    if score9 > best_score:
        best_score = score9
        best_a = a9[:]
        best_b = b9[:]
    
    # If we have time, do local search from the best found
    if time.time() < deadline - 10.0 and best_a is not None:
        random.seed(42)
        current_a = best_a[:]
        current_b = best_b[:]
        current_score = best_score
        
        for _ in range(1000):  # Try up to 1000 local search steps
            if time.time() > deadline - 5.0:
                break
            
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            old_dest = current_a[state] if letter == 0 else current_b[state]
            new_dest = random.randint(0, N - 1)
            
            if letter == 0:
                current_a[state] = new_dest
            else:
                current_b[state] = new_dest
                
            new_score = fast_threshold(current_a, current_b)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                if letter == 0:
                    current_a[state] = old_dest
                else:
                    current_b[state] = old_dest
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
