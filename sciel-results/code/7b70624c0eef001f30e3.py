import time
import numpy as np

N = 17
NUM_BOOLEANS = 1 << N

def _fast_check(network):
    """
    Fast verification using vectorized bitwise operations on all 2^N inputs at once.
    Returns True if the network sorts all inputs.
    """
    if not network:
        return N == 1
    
    # bits[k] is the k-th input pattern (0 to 2^N - 1)
    # We use uint32 to be safe, though N=17 fits in uint16.
    bits = np.arange(NUM_BOOLEANS, dtype=np.uint32)
    
    for (i, j) in network:
        mi = 1 << i
        mj = 1 << j
        # Condition: wire i has 1, wire j has 0
        # (bits & mi) != 0 AND (bits & mj) == 0
        cond = ((bits & mi) != 0) & ((bits & mj) == 0)
        # If condition met, swap i and j.
        # Swapping bits at positions i and j is done by XORing with (mi | mj)
        # because 10 -> 01 and 01 -> 10 are both XOR 11.
        # But we only swap if 10. If 01, we don't swap.
        # So: bits = np.where(cond, bits ^ (mi | mj), bits)
        bits = np.where(cond, bits ^ (mi | mj), bits)
        
    # Check sortedness: No adjacent pair (k, k+1) has 1 followed by 0
    for k in range(N - 1):
        mk = 1 << k
        mk1 = 1 << (k + 1)
        violation = ((bits & mk) != 0) & ((bits & mk1) == 0)
        if np.any(violation):
            return False
            
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Seed network
    net = _odd_even_transposition(N)
    
    # Initial verification
    if not _fast_check(net):
        return net
        
    best_net = list(net)
    best_count = len(net)
    
    # Optimization Loop
    # Strategy: Repeatedly try to remove random comparators.
    # If a removal is successful, restart the process to exploit the new redundancy.
    # This is a form of local search.
    
    # To improve efficiency, we can also try swapping adjacent comparators
    # if removals stall, but removal is the primary driver.
    
    # We use a random shuffle of indices to avoid bias.
    # We also track a "stall" counter to decide when to restart or change strategy.
    
    while time.time() < deadline:
        # Create a random order of comparator indices to try
        indices = list(range(len(net)))
        np.random.shuffle(indices)
        
        removed_any = False
        
        for idx in indices:
            if time.time() > deadline:
                return best_net
            
            # Try removing comparator at idx
            candidate = net[:idx] + net[idx+1:]
            
            if _fast_check(candidate):
                net = candidate
                best_net = list(net)
                best_count = len(net)
                removed_any = True
                # If we removed one, the indices are now stale.
                # Break to re-shuffle and try again from the new state.
                # This ensures we don't try to remove from an invalid state
                # and allows us to find newly exposed redundancies.
                break
        
        if not removed_any:
            # If we couldn't remove anything in a full pass,
            # we might be stuck in a local minimum.
            # Try a few random swaps to perturb the network.
            if len(net) > 1:
                swap_attempts = 0
                while swap_attempts < 10 and time.time() < deadline:
                    i = np.random.randint(0, len(net) - 1)
                    candidate = list(net)
                    candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                    if _fast_check(candidate):
                        # Check if we can remove anything after the swap
                        # (We don't immediately accept the swap unless it helps,
                        # but usually we just accept valid swaps to explore)
                        # Actually, if the swap is valid, we accept it to explore.
                        net = candidate
                        best_net = list(net)
                        break
                    swap_attempts += 1
            else:
                # If we can't remove or swap, we are likely optimal or stuck.
                # Given the time, we might just exit or keep trying removals.
                # But if no removal works in a full pass, and swaps don't help,
                # we are probably done.
                break

    return best_net
# EVOLVE-BLOCK-END
