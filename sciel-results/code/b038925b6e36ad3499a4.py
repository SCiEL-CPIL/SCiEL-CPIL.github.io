import time
import random
import math

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    Optimized with precomputed subset transitions.
    """
    # Precompute transitions for all subsets
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    # Optimization: Use list unpacking or simple loops
    img_a = [1 << x for x in trans_a]
    img_b = [1 << x for x in trans_b]
    
    for subset in range(1, SIZE):
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= img_a[i]
            sb |= img_b[i]
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
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
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin for 90s limit
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Strategy: Deterministic search over known structural families
    # 1. Pinwheel variants: a is a cycle, b maps most states to a "sink" or shift
    # 2. Rank-2 Cycles: specific structures known to have high thresholds
    # 3. Randomized structural mutations around the best found so far
    
    # Helper to evaluate a candidate
    def eval_candidate(a, b):
        nonlocal best_a, best_b, best_len
        l = compute_reset_length(a, b)
        if l != -1 and l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
        return l
    
    # 1. Generate and evaluate Pinwheel variants
    # Standard Pinwheel: a: i -> i+1, b: i -> i-1 (with 0->0 or similar)
    # Actually, the classic Pinwheel is:
    # a: i -> (i+1) % N
    # b: i -> (i-1) % N for i>0, b[0]=0
    # This has threshold (N-1)^2 + N - 1 = 11^2 + 11 = 132 for N=12.
    # Wait, the bound is (N-1)^2 + N - 1. For N=12, 121+11=132.
    # Let's verify the standard pinwheel.
    
    # Variant 1: Standard Pinwheel
    a_pin = [(i + 1) % N for i in range(N)]
    b_pin = [0] * N
    for i in range(1, N):
        b_pin[i] = i - 1
    eval_candidate(a_pin, b_pin)
    
    # Variant 2: Modified Pinwheel (different sink behavior)
    # b: i -> (i-1) % N, but b[0] = N-1? No, that's a cycle.
    # Let's try b: i -> (i-1) % N, b[0]=0 is standard.
    # What if b[0] = 1?
    b_pin2 = [0] * N
    b_pin2[0] = 1
    for i in range(1, N):
        b_pin2[i] = i - 1
    eval_candidate(a_pin, b_pin2)
    
    # Variant 3: Reverse Pinwheel
    a_rev = [(i - 1) % N for i in range(N)]
    b_rev = [0] * N
    for i in range(1, N):
        b_rev[i] = i - 1
    eval_candidate(a_rev, b_rev)
    
    # 2. Rank-2 Cycle Family
    # These are automata where one letter is a permutation (cycle) and the other is a constant or near-constant.
    # Let's try a: cycle, b: constant map to 0.
    # This usually has low threshold.
    
    # 3. Randomized Structural Search
    # We will use the best found so far as a seed for local search.
    # But first, let's do a broader random search over "structured" randoms.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Simulated Annealing / Hill Climbing on the best found
    # We will mutate one transition at a time.
    T = 10.0
    cooling = 0.999
    step = 0
    
    while time.time() < deadline:
        step += 1
        
        # Choose a letter and a state to mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        
        old_val = current_a[state] if letter == 0 else current_b[state]
        
        # Choose a new destination
        new_val = random.randint(0, N - 1)
        while new_val == old_val:
            new_val = random.randint(0, N - 1)
            
        # Apply mutation
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
            
        new_len = compute_reset_length(current_a, current_b)
        
        # If not synchronizing, revert immediately
        if new_len == -1:
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
            continue
            
        # Acceptance criterion
        if new_len > current_len:
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            delta = new_len - current_len
            if delta == 0:
                if random.random() < 0.5:
                    current_len = new_len
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_len = new_len
                else:
                    if letter == 0:
                        current_a[state] = old_val
                    else:
                        current_b[state] = old_val
        
        # Cool down
        T *= cooling
        
        # Restart mechanism: if we haven't improved best for a long time, 
        # we might restart from best or randomize temperature.
        # Simple cooling is usually sufficient for N=12.
        
    return [best_a, best_b]

# Run a quick self-test to ensure it works and is fast
if __name__ == "__main__":
    start = time.time()
    a, b = construct()
    end = time.time()
    l = compute_reset_length(a, b)
    print(f"Time: {end-start:.2f}s, Length: {l}")
    print(f"A: {a}")
    print(f"B: {b}")
