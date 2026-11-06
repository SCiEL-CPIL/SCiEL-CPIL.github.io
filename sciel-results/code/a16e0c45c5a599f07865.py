import time
import random
import sys

N = 17
M = 1 << N  # 131072

# Precompute initial wire states
# Wire w has bit k set iff input k has bit w = 1
# Input k is an integer from 0 to 2^N - 1
# We need 17 integers, each of size 2^N bits

def _init_wire_states():
    states = [0] * N
    for w in range(N):
        val = 0
        # Bit k of val is (k >> w) & 1
        # We can build this by iterating k? No, 131072 iterations is fine in Python if we use shifts?
        # Better: Use the pattern.
        # For w=0: 010101...
        # For w=1: 00110011...
        # Period is 2^(w+1).
        # Within a period of length 2^(w+1), the lower half is 0, upper half is 1.
        # Actually, for bit w:
        # k=0: 0
        # k=1: 0
        # k=2: 1 (if w=1)
        # Let's just use a loop for clarity, it's fast enough for initialization.
        # Or use numpy? No, standard library preferred for portability, but numpy is available.
        # Let's use Python integers.
        
        # Optimization:
        # The pattern repeats every 2^(w+1).
        # Let's construct one period's worth of bits and tile it.
        # But Python integers don't have a "tile" op.
        
        # Simple loop:
        # val = 0
        # for k in range(M):
        #     if (k >> w) & 1:
        #         val |= (1 << k)
        # This is 17 * 131072 = 2.2M iterations. Might take ~0.2s. Acceptable.
        
        # Faster construction:
        # For w=0: 0x5555...
        # For w=1: 0x3333...
        # We can build these using hex strings or bitwise tricks.
        
        # Let's try the loop first, it's safe.
        for k in range(M):
            if (k >> w) & 1:
                val |= (1 << k)
        states[w] = val
    return states

_INIT_STATES = None
def _get_init_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_wire_states()
    return _INIT_STATES

def _verify_fast(network):
    """
    Verify if the network sorts all binary inputs.
    Returns True if valid, False otherwise.
    """
    init = _get_init_states()
    wires = init[:]  # Copy
    
    for (i, j) in network:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max to j
        wires[i] = wi & wj
        wires[j] = wi | wj
        
    # Check sortedness
    # Wire 0 must be 0
    if wires[0] != 0:
        return False
    # Wire N-1 must be all 1s (within M bits)
    mask = (1 << M) - 1
    if wires[N-1] != mask:
        return False
    # For all w, wires[w] must be subset of wires[w+1]
    # i.e., wires[w] & ~wires[w+1] == 0
    for w in range(N - 1):
        if (wires[w] & ~wires[w + 1]) != 0:
            return False
            
    return True

def _seed_odd_even():
    """Odd-even transposition network."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    rng = random.Random(42)
    
    # Start with a good seed
    best_network = _seed_odd_even()
    best_count = len(best_network)
    
    # Verify seed (should be valid)
    if not _verify_fast(best_network):
        # Fallback to a trivially valid but large network if seed fails?
        # Odd-even is always valid.
        pass

    current_network = best_network[:]
    current_count = best_count
    
    def _time_left():
        return deadline - time.time()
        
    # Local Search
    # Strategy:
    # 1. Try to delete a random comparator.
    # 2. If deletion fails repeatedly, try to insert a random comparator.
    # 3. Try to swap adjacent comparators.
    
    stuck_counter = 0
    max_stuck = 100 # If we try 100 deletions and fail, insert one
    
    while _time_left() > 1.0:
        # 1. Attempt Deletion
        deleted = False
        for _ in range(max_stuck):
            if _time_left() < 1.0:
                break
            if not current_network:
                break
            idx = rng.randint(0, len(current_network) - 1)
            trial = current_network[:]
            del trial[idx]
            if _verify_fast(trial):
                current_network = trial
                current_count -= 1
                deleted = True
                stuck_counter = 0
                # Update best if improved
                if current_count < best_count:
                    best_network = current_network[:]
                    best_count = current_count
                break
        
        if deleted:
            continue
            
        # 2. If stuck, try Insertion
        # Insert a random comparator at a random position
        if _time_left() < 1.0:
            break
            
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        pos = rng.randint(0, len(current_network))
        
        trial = current_network[:]
        trial.insert(pos, (i, j))
        
        # We don't strictly need to verify insertion for validity if we just want to explore,
        # but we must ensure we don't get stuck in an invalid state if we rely on validity for search.
        # However, insertion *always* makes the network "more sorted" or at least doesn't break sortedness?
        # No, adding a comparator can break sortedness if it's placed incorrectly?
        # Actually, if a network sorts all inputs, adding a comparator (i,j) anywhere:
        # The new network sorts all inputs IF the comparator is "redundant" or "helpful"?
        # Wait. If Network A sorts all inputs, and we add comparator (i,j) to get Network B.
        # Does B sort all inputs?
        # Let x be an input. A(x) is sorted.
        # B applies A, then (i,j) (if appended).
        # If A(x) is sorted, then A(x)_i <= A(x)_j.
        # Then min(A(x)_i, A(x)_j) = A(x)_i, max = A(x)_j.
        # So the state doesn't change.
        # So appending a comparator to a valid sorting network results in a valid sorting network.
        # What if we insert it in the middle?
        # Network: P, (i,j), Q.
        # Input x. P(x) is some state.
        # (i,j) sorts x_i, x_j.
        # Q sorts the rest.
        # This is NOT guaranteed to be valid.
        # Example: N=3. Network: (0,1). Valid? No.
        # If we have a valid network, inserting a comparator in the middle might break it.
        # So we MUST verify after insertion.
        
        if _verify_fast(trial):
            current_network = trial
            current_count += 1
            # We don't update best_count here because we got worse,
            # but we might enable better deletions later.
            
    return best_network
