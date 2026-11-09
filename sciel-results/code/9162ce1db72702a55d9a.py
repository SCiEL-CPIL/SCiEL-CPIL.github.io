import time
import random

N = 17

# EVOLVE-BLOCK-START

def _verify(net):
    """
    Fast verification using big integers.
    Each wire i is represented by a bit in an integer.
    We simulate all 2^N inputs? No, that's too big.
    Wait, the zero-one principle says we only need to check all 2^N binary inputs.
    2^17 = 131072 inputs.
    Doing this with Python loops is slow.
    Doing this with NumPy is fast but maybe not as fast as bit tricks?
    
    Actually, the inspiration says "Python big-integer bitwise operations, reducing verification time from ~10ms to ~0.1ms".
    How?
    If we represent the state of all 131072 inputs as a set of bitsets?
    Input k (0 to 2^N-1) has bits corresponding to its value.
    Let's represent the network execution for ALL inputs simultaneously.
    
    Let X be a list of N integers. X[i] is a bitmask of length 2^N.
    Bit k of X[i] is 1 if the k-th input has a 1 at wire i.
    Initially, X[i] is the i-th bit of the input index k.
    For input k, wire i is 1 if (k >> i) & 1.
    
    So X[i] = sum(1 << k for k in range(2^N) if (k >> i) & 1)
    
    A comparator (i, j) does:
    min_val = X[i] & X[j]  (Bit is 1 only if both are 1? No. min(1,1)=1, min(1,0)=0, min(0,1)=0, min(0,0)=0. Yes, AND)
    max_val = X[i] | X[j]  (Bit is 1 if either is 1. max(1,1)=1, max(1,0)=1, max(0,1)=1, max(0,0)=0. Yes, OR)
    
    So:
    old_i = X[i]
    old_j = X[j]
    X[i] = old_i & old_j
    X[j] = old_i | old_j
    
    After the network, the sorted output for input k should have:
    Wire 0: 0
    Wire 1: 0
    ...
    Wire (N-1 - popcount(k)): 0
    Wire (N-1 - popcount(k) + 1): 1
    ...
    Wire N-1: 1
    
    So for each input k, let c = popcount(k).
    The output bits should be:
    0 for i < N-1-c
    1 for i >= N-1-c
    
    Let's check if the final X matches this pattern.
    For each wire i, X[i] should be 1 for all k such that popcount(k) <= N-1-i.
    Wait.
    If popcount(k) = c, then there are c ones.
    The ones should be in the highest c wires: N-c, N-c+1, ..., N-1.
    So wire i is 1 iff i >= N-c.
    i >= N-c  <=>  c >= N-i.
    
    So X[i] should have bit k set iff popcount(k) >= N-i.
    
    Let's precompute the target masks.
    target[i] = sum(1 << k for k in range(2^N) if popcount(k) >= N-i)
    
    Then check if X[i] == target[i] for all i.
    """
    size = 1 << N
    # Precompute initial states X
    # X[i] is a bitmask of size 'size'.
    # Bit k is 1 if input k has 1 at wire i.
    # Input k is an integer. Bit i of k is (k >> i) & 1.
    
    # We can construct X[i] efficiently.
    # X[0] has bits 0, 2, 4, ... set? No.
    # k=0: 000...0 -> bit 0 is 0
    # k=1: 000...1 -> bit 0 is 1
    # k=2: 000...10 -> bit 0 is 0
    # So X[0] has odd indices set.
    # X[i] has k set where k & (1 << i) != 0.
    
    # Let's just build them.
    X = [0] * N
    for k in range(size):
        for i in range(N):
            if (k >> i) & 1:
                X[i] |= (1 << k)
    
    # Execute network
    for i, j in net:
        ui = X[i]
        uj = X[j]
        X[i] = ui & uj
        X[j] = ui | uj
    
    # Check against target
    # target[i] has bit k set if popcount(k) >= N - i
    # Let's precompute popcounts or just check on the fly?
    # Precomputing target masks is fast.
    
    # Actually, we can just check the final state.
    # For wire i, it should be 1 for all k with popcount(k) >= N-i.
    
    # Let's precompute popcount for all k?
    # Or just iterate k?
    # Iterating 131072 times in Python is slow (~5-10ms).
    # We want < 1ms.
    
    # Alternative:
    # The final state X[i] must be exactly the mask of all k with popcount(k) >= N-i.
    # Let's precompute these target masks once outside the function?
    # Yes, global cache.
    
    global _TARGETS
    if '_TARGETS' not in globals():
        _TARGETS = [0] * N
        # Count popcounts
        popcounts = [0] * size
        for k in range(size):
            # popcount(k)
            # Use bin(k).count('1')? Slow.
            # Use lookup or bit trick?
            # Python 3.8+ has int.bit_count()
            popcounts[k] = k.bit_count()
        
        for i in range(N):
            threshold = N - i
            mask = 0
            for k in range(size):
                if popcounts[k] >= threshold:
                    mask |= (1 << k)
            _TARGETS[i] = mask
    
    for i in range(N):
        if X[i] != _TARGETS[i]:
            return False
    return True

def _get_seed():
    # Odd-even transposition
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with a valid network
    # Using a slightly better seed if possible? 
    # Odd-even is 136.
    # Let's try to improve it.
    
    current_net = _get_seed()
    best_net = list(current_net)
    best_len = len(current_net)
    
    # If we can't verify fast, this won't work.
    # Let's test verification speed first.
    t0 = time.time()
    _verify(current_net)
    t1 = time.time()
    verify_time = t1 - t0
    # print(f"Verify time: {verify_time}")
    
    # If verify is too slow, we might need to reduce search space or use a different method.
    # But the inspiration says 0.1ms.
    # 85s / 0.1ms = 850,000 steps. Plenty.
    
    # Simulated Annealing / Random Restart Local Search
    # Moves:
    # 1. Remove a comparator (if valid)
    # 2. Swap two comparators
    # 3. Add a random comparator (if valid? or accept if better?)
    #    Usually, we want to minimize size.
    #    So:
    #    - Try removing a random comparator. If valid, accept.
    #    - Try swapping two comparators. If valid, accept (size same, but might enable further removals).
    #    - Try moving a comparator?
    
    # Strategy:
    # Loop:
    #   - Pick a random move.
    #   - If move reduces size (removal), verify. If valid, accept.
    #   - If move keeps size (swap), verify. If valid, accept with some probability (to escape local minima).
    #   - If move increases size (addition), rarely accept.
    
    # Actually, simple greedy removal + random swaps is often good.
    
    random.seed(42)
    
    # Precompute all possible pairs?
    # N=17, pairs = 17*16/2 = 136.
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Choose a move
        # 70% removal attempt, 30% swap attempt
        if random.random() < 0.7:
            # Try to remove a random comparator
            if not current_net:
                break
            idx = random.randint(0, len(current_net) - 1)
            candidate = list(current_net)
            candidate.pop(idx)
            
            if _verify(candidate):
                current_net = candidate
                best_net = list(current_net)
                best_len = len(current_net)
                # print(f"Removed! Size: {best_len}")
        else:
            # Try to swap two comparators
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            # Only verify if we think it's worth it?
            # Swaps don't change size, but might allow future removals.
            # We should accept swaps that maintain validity.
            if _verify(candidate):
                # Accept swap with some probability to allow exploration?
                # Or always accept?
                # If we always accept, we might wander.
                # Let's accept with prob 0.5 to keep some structure?
                # Or just accept.
                current_net = candidate
        
        # Periodic restart?
        # If stuck for too long, restart from best_net with some noise?
        # For now, keep it simple.
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
            
    return best_net

# EVOLVE-BLOCK-END
