"""Černý problem: 12-state binary automata with long reset thresholds."""
import time
import random

N = 12
DEADLINE = 85.0  # Safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Helper: compute exact shortest reset word length via BFS on subset automaton
    # Returns (length, is_sync)
    def reset_length(a, b):
        # Precompute single-state transitions as bitmasks
        # For letter a: a_mask[i] = 1 << a[i]
        # For letter b: b_mask[i] = 1 << b[i]
        a_mask = [1 << x for x in a]
        b_mask = [1 << x for x in b]
        
        full = (1 << N) - 1  # bitmask of all states
        
        # BFS from full set
        # State: subset bitmask
        # We want to reach a singleton (exactly one bit set)
        
        # Precompute transition of any subset under a and b
        # For efficiency, we can compute on the fly or precompute all 2^N subsets
        # 2^12 = 4096 subsets, each transition is OR of up to 12 single-state masks
        
        # Let's precompute trans_a[subset] and trans_b[subset]
        # But that's 4096 * 2 = 8192 entries, each requiring OR of set bits
        # Actually, we can compute trans for a subset using DP or just iterate
        
        # Simpler: BFS with lazy transition computation
        # Use a queue: (subset, depth)
        # Visited: set of subsets
        
        from collections import deque
        
        visited = set()
        visited.add(full)
        queue = deque()
        queue.append((full, 0))
        
        # Check if full is already a singleton (N=1 case, not our case)
        if full & (full - 1) == 0:
            return (0, True)
        
        while queue:
            subset, depth = queue.popleft()
            
            # Transition under 'a'
            new_subset_a = 0
            s = subset
            while s:
                i = (s & -s).bit_length() - 1
                new_subset_a |= a_mask[i]
                s &= s - 1
            if new_subset_a not in visited:
                if new_subset_a & (new_subset_a - 1) == 0:
                    return (depth + 1, True)
                visited.add(new_subset_a)
                queue.append((new_subset_a, depth + 1))
            
            # Transition under 'b'
            new_subset_b = 0
            s = subset
            while s:
                i = (s & -s).bit_length() - 1
                new_subset_b |= b_mask[i]
                s &= s - 1
            if new_subset_b not in visited:
                if new_subset_b & (new_subset_b - 1) == 0:
                    return (depth + 1, True)
                visited.add(new_subset_b)
                queue.append((new_subset_b, depth + 1))
        
        return (-1, False)  # Not synchronizing
    
    # Helper: generate a random automaton
    def random_automaton():
        a = [random.randrange(N) for _ in range(N)]
        b = [random.randrange(N) for _ in range(N)]
        return a, b
    
    # Known good constructions to start with
    candidates = []
    
    # 1. Simple cycle + reset (the baseline)
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    candidates.append((a, b))
    
    # 2. Pinwheel-like: a is a full cycle, b maps most to 0 but one exception
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[N-1] = N-2  # slight modification
    candidates.append((a, b))
    
    # 3. Another variant: a is cycle, b maps i to i-1 except 0->0
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i-1 for i in range(1, N)]
    candidates.append((a, b))
    
    # 4. a: i -> i+1, b: i -> (i+1)%N for i<11, b[11]=0
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)] + [0]
    candidates.append((a, b))
    
    # 5. a: permutation, b: non-permutation with good mixing
    import random as rnd
    rnd.seed(42)
    a = list(range(N))
    rnd.shuffle(a)
    b = [0] * N
    candidates.append((a, b))
    
    # Evaluate initial candidates
    best_a, best_b = None, None
    best_len = -1
    
    for a, b in candidates:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        length, sync = reset_length(a, b)
        if sync and length > best_len:
            best_len = length
            best_a, best_b = a, b
    
    # Random search phase
    # For N=12, reset_length BFS visits at most 2^12 = 4096 states
    # Each state requires O(N) work for transition computation
    # So each evaluation is O(4096 * 12) ≈ 50k operations
    # We can do thousands of evaluations in 85 seconds
    
    random.seed(12345)
    
    trial = 0
    while time.time() - start_time < DEADLINE - 2:
        a, b = random_automaton()
        length, sync = reset_length(a, b)
        if sync and length > best_len:
            best_len = length
            best_a, best_b = a, b
        trial += 1
        
        # Every 100 trials, check if we're making progress
        if trial % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
    
    if best_a is None:
        # Fallback: return the simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
