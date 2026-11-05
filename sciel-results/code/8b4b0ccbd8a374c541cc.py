import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096

# Precompute all possible binary inputs (0 to 2^16 - 1)
# Represent each input as a 16-bit integer.
# Bit k is 1 if wire k has value 1.

def _verify_network_full(network):
    """
    Verify if the network sorts all 2^16 inputs.
    Returns True if valid, False otherwise.
    Uses vectorized NumPy operations for speed.
    """
    if not network:
        return False
        
    states = np.arange(2**16, dtype=np.uint32)
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        # If bit_i is 1 and bit_j is 0, we need to swap.
        # swap_mask is 1 where bit_i=1 and bit_j=0.
        swap_mask = bit_i & (~bit_j) & 1
        # Apply swap: XOR with (1<<i) | (1<<j)
        # But only where swap_mask is 1.
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    # Check if all states are sorted.
    # Sorted state: 0...01...1.
    # This means the set of 0-bits is a prefix and 1-bits is a suffix.
    # Let x = ~state. The 0-bits of state are 1-bits of x.
    # x must be of the form 2^m - 1 (contiguous 1s at LSB).
    # This is true iff x & (x + 1) == 0.
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def _fallback_construct():
    """
    Returns the standard Odd-Even Transposition network.
    This is a valid sorting network with 120 comparators.
    """
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Initialize with a known valid network
    # Odd-Even Transposition is 120 comparators.
    best_network = _fallback_construct()
    best_score = len(best_network)
    
    if not _verify_network_full(best_network):
        # This should never happen, but safety first
        return best_network

    # 2. Simulated Annealing
    # We want to minimize the number of comparators.
    # Moves:
    #   - Remove a comparator (if valid, always accept; if invalid, reject)
    #   - Swap two comparators (reordering, might help future removals)
    #   - Replace a comparator with another (might help escape local minima)
    
    # For efficiency, we primarily focus on REMOVAL and REORDERING.
    # Replacement is expensive to verify if it breaks validity, but we can use a heuristic.
    # Actually, standard local search for sorting networks often uses:
    #   1. Try to remove any comparator. If valid, keep it.
    #   2. If stuck, try swapping adjacent comparators (if they commute or just reorder).
    #   3. If still stuck, try "mutating" a comparator (changing i or j) and see if we can recover validity with fewer ops later.
    
    # A more robust SA approach for this specific problem (minimizing size):
    # State: List of comparators.
    # Cost: Length of list.
    # But we MUST maintain validity.
    # So it's a constrained optimization.
    
    # Strategy:
    # 1. Aggressive Removal: Try to remove every comparator. If removal keeps it valid, do it.
    # 2. If no removal is possible, try to "reorder" the network. 
    #    Reordering doesn't change size, but might allow further removals.
    #    However, reordering a valid network usually keeps it valid only if the comparators are "compatible" (don't interfere).
    #    Actually, ANY permutation of a valid sorting network is valid? NO.
    #    Sorting networks are directed acyclic graphs effectively. You can't just shuffle comparators arbitrarily.
    #    However, comparators on disjoint wires can be swapped.
    #    Comparators sharing wires cannot be arbitrarily swapped.
    
    # Let's stick to a simpler, proven heuristic for this task type:
    # "Random Restart Local Search" with a specific focus on removal.
    # Since we start with 120, we try to remove.
    # If we get stuck (no single removal works), we can try to "perturb" the network by:
    #   - Removing a comparator (invalid)
    #   - Adding a DIFFERENT comparator (increases size, but might open a path to remove 2 later?)
    #   This is risky and time-consuming.
    
    # Alternative: Use the "Zero-One Principle" to guide removal.
    # We can track which inputs are "critical" for each comparator.
    
    # Let's implement a multi-start random local search:
    # 1. Start with 120.
    # 2. Loop:
    #    a. Try to remove a random comparator. If valid, keep it.
    #    b. If no removal works after K attempts, try to swap two comparators that don't share wires (commutable).
    #       This reorders the network. It doesn't change size, but changes the "shape".
    #       Does reordering help removal? Yes, if the network was in a "bad" order for removal.
    #    c. If stuck, try a "move" that changes the network structure slightly:
    #       Replace comparator (i,j) with (i', j')? 
    #       This is hard to verify quickly.
    
    # Let's try a different structural change:
    # Instead of just removing, we perform a "swap" of comparator endpoints if it preserves validity?
    # No, changing (i,j) to (i,k) changes the logic.
    
    # Let's rely on the finding from "inspiration 2" which used `try_remove`, `try_insert`.
    # The key insight from the "1.5 score" observation is that bit-parallel verification allows many steps.
    
    # Implementation:
    # 1. Start with best_network.
    # 2. While time remains:
    #    - Pick a random comparator index k.
    #    - Try removing network[k].
    #    - If valid, remove it. Update best.
    #    - If invalid, try swapping network[k] with a random other comparator network[m] 
    #      IF they are commutable (disjoint wires).
    #      Commutability: (i1, j1) and (i2, j2) are commutable if {i1, j1} and {i2, j2} are disjoint.
    #      If we swap them, the network remains valid? 
    #      Actually, if they are disjoint, their order doesn't matter. So the network is still valid.
    #      This allows us to explore different orderings of disjoint comparators.
    #      Does this help? It might allow a comparator that was "blocked" by a non-commutable one to be removed?
    #      No, removal depends on the *set* of comparators, not their order, IF the network is a DAG.
    #      Wait, sorting networks are NOT just sets. The order matters.
    #      Comparator (0,1) then (1,2) is different from (1,2) then (0,1).
    #      So reordering DOES change the function.
    #      If we swap two DISJOINT comparators, the function is IDENTICAL.
    #      So swapping disjoint comparators is a "no-op" in terms of function, but changes the list representation.
    #      This doesn't help escape local minima in the space of *functions*.
    #      It only helps if our removal heuristic depends on position? No, removal is global.
    
    # So, to escape local minima, we must CHANGE the function.
    # Move: Replace comparator (i, j) with (i, j') or (i', j).
    # This changes the function. It might make the network invalid.
    # If invalid, we reject.
    # If valid, we accept (size is same).
    # Then we try to remove.
    
    # This is "Random Walk on Valid Networks".
    
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    comparator_set = set(all_comparators)
    
    current_network = best_network[:]
    
    while time.time() < deadline:
        # 1. Try to remove
        removed = False
        # Try a few random removals
        for _ in range(10):
            if not current_network:
                break
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx+1:]
            if _verify_network_full(candidate):
                current_network = candidate
                if len(current_network) < best_score:
                    best_network = current_network[:]
                    best_score = len(current_network)
                removed = True
                break
        
        if removed:
            continue
            
        # 2. If stuck, try to mutate (replace a comparator) to escape local minimum
        # We want to change the network so that a removal becomes possible.
        # Pick a random comparator in current_network.
        if current_network:
            idx = random.randint(0, len(current_network) - 1)
            i, j = current_network[idx]
            
            # Try to replace (i, j) with a random other comparator
            # To keep the network "close", maybe pick a comparator that shares a wire?
            # Or just random.
            # Let's try random replacement.
            # We need to ensure the new comparator is not already in the network?
            # Actually, duplicate comparators are allowed but useless.
            # Let's pick a random comparator from all_comparators.
            new_i, new_j = random.choice(all_comparators)
            
            if (new_i, new_j) == (i, j):
                continue
                
            candidate = current_network[:]
            candidate[idx] = (new_i, new_j)
            
            # Verify candidate
            if _verify_network_full(candidate):
                current_network = candidate
                # Size is same, so no update to best_score yet.
                # But we are in a new state.
                
    return best_network

# Keep the global N and fallback for interface compatibility
N = 16

def _fallback_construct():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# Re-define construct to be the main entry point
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    try:
        # Start with valid baseline
        net = _fallback_construct()
        
        if not _verify_network_full(net):
            return net
            
        best_net = net[:]
        best_len = len(net)
        
        all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        while time.time() < deadline:
            # Phase 1: Greedy Removal
            progress = False
            # Try to remove any comparator
            # To be fast, we don't check all, we check random ones
            # But if we are close to optimal, we might need to check all.
            # Let's try a mix:
            # 1. Try random removals
            # 2. If no progress, try all removals (if network is small enough)
            
            for _ in range(5):
                if not net:
                    break
                idx = random.randint(0, len(net) - 1)
                cand = net[:idx] + net[idx+1:]
                if _verify_network_full(cand):
                    net = cand
                    if len(net) < best_len:
                        best_net = net[:]
                        best_len = len(net)
                    progress = True
                    break
            
            if progress:
                continue
                
            # Phase 2: Mutation (if stuck)
            # Try to change a comparator to a different one
            # This explores the space of valid networks of the same size
            if net:
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                
                # Pick a new comparator
                # Bias towards "similar" comparators?
                # Just random for now.
                new_i, new_j = random.choice(all_comparators)
                
                if (new_i, new_j) == (i, j):
                    continue
                    
                cand = net[:]
                cand[idx] = (new_i, new_j)
                
                if _verify_network_full(cand):
                    net = cand
                    
        return best_net
        
    except Exception:
        return _fallback_construct()

# Ensure imports are at the top for the final block
import time
import random
import numpy as np
