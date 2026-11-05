"""
Construct a 12-state binary synchronizing automaton maximizing the shortest reset word length.

Strategy:
  1. Fast bitmask-based BFS to compute exact reset threshold in <1ms.
  2. Random restart local search:
       - Generate random complete deterministic automata.
       - If synchronizing, compute threshold.
       - Local search: mutate one transition (random state, random letter, new target).
       - Accept if still synchronizing and threshold increases (or equal with some probability).
  3. Deadline guard: track elapsed time, return best found before 85s.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1  # 0xFFF for N=12

def _reset_threshold(trans_a, trans_b, timeout=None):
    """
    Compute the exact shortest reset word length for a binary deterministic automaton.
    trans_a, trans_b: lists of length N, trans_x[i] = destination state for state i on letter x.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    Uses bitmask representation: a subset S is represented as a bitmask.
    BFS from the full set {0,...,N-1} (mask = FULL_MASK) to any singleton.
    """
    # Precompute transition tables as dicts: mask -> new_mask for each letter
    # But since N=12, masks are up to 4096, we can precompute on the fly.
    
    # For speed, precompute the effect of each letter on each possible mask.
    # There are 2^12 = 4096 possible masks.
    # trans_a_mask[mask] = new mask after applying letter 'a' to subset represented by mask.
    
    def build_trans_table(trans):
        table = [0] * (1 << N)
        for mask in range(1 << N):
            new_mask = 0
            for i in range(N):
                if mask & (1 << i):
                    new_mask |= (1 << trans[i])
            table[mask] = new_mask
        return table
    
    ta = build_trans_table(trans_a)
    tb = build_trans_table(trans_b)
    
    # BFS
    # State: mask
    # We want to reach any singleton mask (exactly one bit set).
    # Singletons: 1, 2, 4, ..., 1<<11
    singleton_set = set(1 << i for i in range(N))
    
    start = FULL_MASK
    if start in singleton_set:
        return 0  # Already a singleton (impossible for N>1, but safe)
    
    visited = [False] * (1 << N)
    visited[start] = True
    queue = deque()
    queue.append((start, 0))  # (mask, depth)
    
    while queue:
        mask, depth = queue.popleft()
        
        # Try both letters
        for new_mask in (ta[mask], tb[mask]):
            if new_mask in singleton_set:
                return depth + 1
            if not visited[new_mask]:
                visited[new_mask] = True
                queue.append((new_mask, depth + 1))
    
    return -1  # Not synchronizing


def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    best_threshold = -1
    best_a = None
    best_b = None
    
    # Seed some initial candidates with known good structures
    # The seed program: cycle + reset
    seed_a = [(i + 1) % N for i in range(N)]
    seed_b = [0] * N
    thr = _reset_threshold(seed_a, seed_b)
    if thr > best_threshold:
        best_threshold = thr
        best_a = seed_a[:]
        best_b = seed_b[:]
    
    # Random restart local search
    rng = random.Random(42)
    
    max_local_steps = 2000  # Max mutations per restart
    
    while time.time() < deadline:
        # Check deadline before starting a new restart
        if time.time() > deadline:
            break
        
        # Generate a random initial automaton
        # Strategy: bias towards structures that tend to synchronize
        # Option 1: Random complete deterministic automaton
        # Option 2: Cycle + random perturbations
        # Option 3: One letter is a permutation (cycle), other is random
        
        choice = rng.random()
        if choice < 0.3:
            # One letter is a cycle, other is random
            a = list(range(N))
            rng.shuffle(a)
            b = [rng.randint(0, N-1) for _ in range(N)]
        elif choice < 0.6:
            # Both random, but ensure at least one state has both letters going to same place (helps sync)
            a = [rng.randint(0, N-1) for _ in range(N)]
            b = [rng.randint(0, N-1) for _ in range(N)]
            # Force one synchronization point
            s = rng.randint(0, N-1)
            b[s] = a[s]
        else:
            # Fully random
            a = [rng.randint(0, N-1) for _ in range(N)]
            b = [rng.randint(0, N-1) for _ in range(N)]
        
        cur_a = a[:]
        cur_b = b[:]
        cur_thr = _reset_threshold(cur_a, cur_b)
        
        if cur_thr < 0:
            continue
        
        if cur_thr > best_threshold:
            best_threshold = cur_thr
            best_a = cur_a[:]
            best_b = cur_b[:]
        
        # Local search
        local_steps = 0
        while local_steps < max_local_steps and time.time() < deadline:
            local_steps += 1
            
            # Mutate one transition
            letter = rng.randint(0, 1)
            state = rng.randint(0, N-1)
            new_target = rng.randint(0, N-1)
            
            if letter == 0:
                old_target = cur_a[state]
                if old_target == new_target:
                    continue
                cur_a[state] = new_target
                new_thr = _reset_threshold(cur_a, cur_b)
                cur_a[state] = old_target  # revert
            else:
                old_target = cur_b[state]
                if old_target == new_target:
                    continue
                cur_b[state] = new_target
                new_thr = _reset_threshold(cur_a, cur_b)
                cur_b[state] = old_target  # revert
            
            if new_thr < 0:
                continue
            
            # Accept if better, or equal with small probability (to escape plateaus)
            if new_thr > cur_thr:
                # Apply the mutation
                if letter == 0:
                    cur_a[state] = new_target
                else:
                    cur_b[state] = new_target
                cur_thr = new_thr
                
                if cur_thr > best_threshold:
                    best_threshold = cur_thr
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            elif new_thr == cur_thr and rng.random() < 0.1:
                # Accept equal to explore
                if letter == 0:
                    cur_a[state] = new_target
                else:
                    cur_b[state] = new_target
                cur_thr = new_thr
        
        # Occasional restart with a different bias
        if best_threshold >= 20:
            # We're doing well, try more targeted searches
            pass
    
    # Return the best found
    if best_a is None:
        # Fallback to seed
        return seed_a, seed_b
    
    return best_a, best_b
