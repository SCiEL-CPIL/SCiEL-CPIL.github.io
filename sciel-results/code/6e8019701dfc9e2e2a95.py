import time
import random
import numpy as np

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    Optimized for speed using numpy where possible, but pure python bitmask BFS is often faster for N=12.
    """
    size = SIZE
    # Precompute transitions for all subsets
    # We can optimize this by precomputing the mapping for each state and letter
    # But for N=12, 4096 subsets * 12 states is small.
    
    # To make it faster, we can precompute the subset transitions for a and b
    # trans_a_sub[subset] = subset of destinations when applying 'a' to all states in subset
    
    # Actually, a faster BFS implementation for small N:
    # 1. Precompute next_a[subset] and next_b[subset]
    # 2. BFS from (1<<N)-1
    
    # Let's try to speed up the precomputation.
    # For a given transition function f (list of N ints), 
    # next_f[subset] = OR (1 << f[i] for i in subset)
    
    # We can compute next_f for all subsets using DP or just direct calculation.
    # Direct calculation:
    # for subset in range(SIZE):
    #    next = 0
    #    for i in range(N):
    #        if subset & (1 << i):
    #            next |= 1 << f[i]
    
    # This is 4096 * 12 = 49k operations. Very fast.
    
    next_a = [0] * size
    next_b = [0] * size
    
    # Precompute masks for states
    state_masks = [1 << i for i in range(N)]
    
    for subset in range(size):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        # Iterate through set bits
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= state_masks[trans_a[i]]
            sb |= state_masks[trans_b[i]]
            temp ^= lsb
        next_a[subset] = sa
        next_b[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            next_a_subset = next_a[subset]
            if not visited[next_a_subset]:
                visited[next_a_subset] = True
                new_queue.append(next_a_subset)
            
            next_b_subset = next_b[subset]
            if not visited[next_b_subset]:
                visited[next_b_subset] = True
                new_queue.append(next_b_subset)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # 1. Initialize with a known high-threshold automaton
    # The Pinwheel automaton P_n is a good starting point.
    # For N=12, the standard Pinwheel often has a threshold around 110-112.
    # Let's construct a specific Pinwheel variant known to be strong.
    
    # Structure:
    # a: (0 1 2 ... N-1)
    # b: (0 0 1 ... N-2) is not quite right.
    # A common high-threshold family is:
    # a[i] = (i + 1) % N
    # b[i] = (i - 1) % N for i > 0, b[0] = 0 ? No, that's a star.
    
    # Let's use the "Rank-2" or "Pinwheel" construction from literature for N=12.
    # One known optimal for N=12 has length 121.
    # We don't have the exact structure memorized, but we can start with a strong candidate.
    
    # Candidate 1: Pinwheel
    # a = [1, 2, ..., N-1, 0]
    # b = [0, 0, 1, ..., N-2] -> This is a "comb".
    # Let's try a few initial candidates and pick the best.
    
    candidates = []
    
    # Candidate A: Standard Pinwheel
    a_a = [(i + 1) % N for i in range(N)]
    b_a = [0] * N
    for i in range(1, N):
        b_a[i] = i - 1
    candidates.append((a_a, b_a))
    
    # Candidate B: Reverse Pinwheel
    a_b = [(i - 1) % N for i in range(N)]
    b_b = [0] * N
    for i in range(1, N):
        b_b[i] = i - 1
    candidates.append((a_b, b_b))
    
    # Candidate C: "Shift" with break
    a_c = [(i + 1) % N for i in range(N)]
    b_c = [(i - 1) % N for i in range(N)]
    b_c[0] = 0 # Break at 0
    candidates.append((a_c, b_c))
    
    # Evaluate candidates to find the best start
    best_a = None
    best_b = None
    best_len = -1
    
    for a, b in candidates:
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    # 2. Stochastic Hill Climbing
    # We will mutate one transition entry at a time.
    # If the new length is >= best_len, accept it.
    # If > best_len, update best.
    # We use a "plateau" strategy: if we don't improve for X steps, we might
    # accept a neutral move to escape local minima, or restart from a random point.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    no_improve_count = 0
    max_no_improve = 1000  # If stuck for 1000 steps, try a random restart or larger jump
    
    while time.time() < deadline:
        # Pick a random letter and state to mutate
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        
        # Save old value
        old_val = current_a[state] if letter == 0 else current_b[state]
        
        # Generate new value (randomly different from old)
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            new_val = (new_val + 1) % N
            
        # Apply mutation
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
            
        # Evaluate
        new_len = compute_reset_length(current_a, current_b)
        
        if new_len >= current_len:
            # Accept
            current_len = new_len
            no_improve_count = 0
            if new_len > best_len:
                best_len = new_len
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Reject, revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
            no_improve_count += 1
            
        # If stuck for too long, try a "kick"
        # A kick is a random mutation that is accepted regardless of score,
        # but we revert if it drops the score too much? 
        # Or just restart from the best known?
        if no_improve_count > max_no_improve:
            # Restart from best known to ensure we are on a good plateau
            current_a = best_a[:]
            current_b = best_b[:]
            current_len = best_len
            no_improve_count = 0
            
            # Optional: Add a random perturbation to the best known
            # to explore a new neighborhood
            for _ in range(5):
                l = random.choice([0, 1])
                s = random.randint(0, N - 1)
                if l == 0:
                    current_a[s] = random.randint(0, N - 1)
                else:
                    current_b[s] = random.randint(0, N - 1)
            
            # Evaluate the perturbed best
            perturbed_len = compute_reset_length(current_a, current_b)
            if perturbed_len >= best_len:
                current_len = perturbed_len
            else:
                # Revert to best
                current_a = best_a[:]
                current_b = best_b[:]
                current_len = best_len

    return [best_a, best_b]
