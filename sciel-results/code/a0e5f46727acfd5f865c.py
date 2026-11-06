import time
import random
import itertools

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    Optimized with precomputed subset transitions.
    """
    # Precompute transition for each subset for letter a and b
    # We can compute these on the fly or precompute.
    # Precomputation cost: 4096 subsets * 2 letters * (popcount operations)
    
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    # Optimize precomputation by iterating bits
    for subset in range(SIZE):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = SIZE - 1
    if start == 1: # Only state 0
        return 0
        
    visited = [False] * SIZE
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            # Apply a
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            # Apply b
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 85.0 # Safety margin (90s limit, 5s buffer)
    
    best_len = -1
    best_a = None
    best_b = None
    
    # Strategy: Focus on the "Pinwheel" structure.
    # a is a cycle: a[i] = (i+1) % N
    # b is a permutation.
    # It is known that for the Pinwheel automaton (a is cycle, b is a specific permutation),
    # the reset length can be high.
    # Specifically, the "Cherny" construction often involves a and b being permutations
    # or near-permutations.
    
    # Let's fix a as the cycle: a[i] = (i+1) % N
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # We will search for b among permutations of 0..N-1.
    # Total permutations: 12! = 479,001,600. Too many to exhaust.
    # However, we can use symmetries or specific structures.
    
    # A known high-threshold structure for N=12 is related to the "pinwheel" where
    # b maps i to i-1 (mod N) except for one state, or similar.
    # Actually, the maximum reset length for N=12 is known to be 55 (or close to it? 
    # Cherny bound is (N-1)^2 = 121. 
    # For N=12, the best known is 55? Let's check small N.
    # N=2: 1
    # N=3: 3
    # N=4: 6
    # N=5: 10
    # N=6: 15
    # N=7: 21
    # N=8: 28
    # N=9: 36
    # N=10: 45
    # N=11: 55
    # N=12: 66? (Triangular numbers T_{N-1} = (N-1)N/2? No, that's for specific cases).
    # Actually, the pinwheel automaton with a=(1 2 ... N) and b=(1 2 ... N-1) (fixing N) 
    # has reset length N-1.
    # The "binary" automata with max reset length are often constructed by "rotating" the sink.
    
    # Let's use a smarter search:
    # 1. Try the "standard" pinwheel: a is cycle, b is cycle with one fixed point?
    #    b[i] = i-1 for i>0, b[0]=0? No.
    #    
    # Let's try a randomized local search on the space of permutations for b, 
    # starting from the "reverse cycle" or "shift" permutations.
    # Also, we can vary a slightly if needed, but a=cycle is a strong candidate.
    
    # Candidate 1: a = cycle, b = reverse cycle (b[i] = (N-1-i) % N)
    # Candidate 2: a = cycle, b = shift (b[i] = (i-1) % N)
    # Candidate 3: a = cycle, b = random permutation
    
    # Let's run a local search on permutations of b.
    # We represent b as a list.
    
    # Initialize with a few strong candidates
    candidates = []
    
    # 1. b is identity (b[i]=i) -> a is cycle. Reset length? 
    #    If b is identity, we can just use a^N. Length N. Not great.
    
    # 2. b is reverse: b[i] = (N-1-i) % N
    b_rev = [(N - 1 - i) % N for i in range(N)]
    candidates.append(b_rev)
    
    # 3. b is shift left: b[i] = (i-1) % N
    b_shift = [(i - 1) % N for i in range(N)]
    candidates.append(b_shift)
    
    # 4. Random permutations
    for _ in range(20):
        b_rand = list(range(N))
        random.shuffle(b_rand)
        candidates.append(b_rand)
        
    # Evaluate initial candidates
    for b in candidates:
        l = compute_reset_length(a_cycle, b)
        if l > best_len:
            best_len = l
            best_a = a_cycle[:]
            best_b = b[:]
            
    # Local Search on b (permutations)
    # We will perform hill climbing on the permutation b.
    # Move: swap two elements in b, or change b[i] to a random value (if we allow non-permutations).
    # Wait, b does NOT have to be a permutation. It just has to be a function 0..N-1 -> 0..N-1.
    # However, if b is not a permutation, it might synchronize faster.
    # To maximize reset length, we often want b to be "as close to a permutation as possible" 
    # but with a "sink" structure that allows slow collapse.
    
    # Actually, the maximum reset length automata are often "pinwheels" where 
    # a is a cycle and b is a "near-permutation" that maps most states to previous ones 
    # but has a specific structure.
    
    # Let's allow b to be any function, but start from permutations and mutate.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Choose a mutation
        # 70% chance to mutate b, 30% chance to mutate a (less likely to help if a is cycle, but possible)
        if random.random() < 0.7:
            i = random.randint(0, N-1)
            old_val = current_b[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_b[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                current_b[i] = old_val
                no_improve += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve = 0
            else:
                current_b[i] = old_val
                no_improve += 1
        else:
            i = random.randint(0, N-1)
            old_val = current_a[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_a[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                current_a[i] = old_val
                no_improve += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve = 0
            else:
                current_a[i] = old_val
                no_improve += 1
        
        # Restart if stuck
        if no_improve > max_no_improve:
            # Revert to best
            current_a = best_a[:]
            current_b = best_b[:]
            current_len = best_len
            no_improve = 0
            
            # Try a different random restart
            if random.random() < 0.5:
                # Restart with random b
                b_rand = list(range(N))
                random.shuffle(b_rand)
                current_b = b_rand
                l = compute_reset_length(current_a, current_b)
                if l != -1:
                    current_len = l
            else:
                # Restart with random a and b
                a_rand = [random.randint(0, N-1) for _ in range(N)]
                b_rand = [random.randint(0, N-1) for _ in range(N)]
                l = compute_reset_length(a_rand, b_rand)
                if l != -1:
                    current_a = a_rand
                    current_b = b_rand
                    current_len = l
                else:
                    # Revert
                    current_a = best_a[:]
                    current_b = best_b[:]
                    current_len = best_len

    return [best_a, best_b]
