import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

# Pre-compute initial and expected wires for verification
# This is done once at module load time to ensure it's ready for construct()
_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_layered_network(depth_target=None):
    """
    Constructs a network using a layered approach (like Bitonic or Brent-Kung simplified).
    This creates a structurally different starting point than the seed.
    We aim for a depth that is feasible, then optimize size.
    """
    if depth_target is None:
        # Heuristic depth for 17 wires. 
        # Brent-Kung for 16 wires is depth 12. 
        # Let's try to construct a network with depth ~12-14 first.
        depth_target = 12
        
    net = []
    # A simple layered construction:
    # Phase 1: Sort pairs (0,1), (2,3)...
    # Phase 2: Sort (1,2), (3,4)...
    # ... this is just odd-even.
    
    # Instead, let's use a recursive "sort network" builder that ensures correctness
    # but might not be optimal. 
    # Actually, the seed is already a valid network. 
    # The goal of this strategy is to start from a *different* valid network 
    # that might have a different "shape" allowing for better reduction.
    
    # Let's try a "Bitonic" style construction.
    # Bitonic sort network for 17 wires? Bitonic requires power of 2.
    # We can pad to 32? No, that increases size.
    
    # Alternative: Use the seed, but then immediately apply a "random shuffle" 
    # of the comparator order that preserves validity? 
    # No, shuffling usually breaks validity.
    
    # Better structural change: 
    # Start with the seed. 
    # The current program does greedy removal. 
    # This program will do:
    # 1. Greedy removal to get to a local minimum.
    # 2. "Simulated Annealing" on the set of comparators:
    #    - Allow adding a comparator (increasing size) to escape local minima.
    #    - Allow removing a comparator (decreasing size).
    #    - Accept worse solutions with probability based on "temperature".
    #    - Track the best solution found.
    
    # However, the prompt asks for a MATERIALLY DIFFERENT strategy.
    # Greedy removal is the current strategy.
    # Simulated Annealing with insertions is different.
    
    # Let's implement a local search that allows:
    # 1. Remove random comparator (if valid, keep; if not, revert)
    # 2. Add random comparator (if valid, keep; if not, revert) -- but we only keep if it leads to better future removals? 
    #    Actually, adding a comparator increases size, which is bad for score.
    #    But it might allow removing 2 others later.
    #    So we accept additions temporarily if they are part of a "move".
    
    # Simpler different strategy: 
    # **Genetic Algorithm / Crossover**
    # Maintain a population of valid networks.
    # Crossover: Take first half of net A and second half of net B. Verify.
    # Mutation: Add/Remove/Swap comparator. Verify.
    # Selection: Keep smallest valid networks.
    
    # Given the 60s limit and 131k verification cost, GA might be too slow 
    # unless verification is very fast. 
    # Verification is ~1-5ms. 
    # 10,000 verifications in 30s. 
    # GA population 10, generations 1000? 
    # 10 * 1000 * 2 (crossover+mutation) = 20,000 evaluations. 
    # Might be tight but possible.
    
    # Let's stick to a **Hill Climbing with Random Restarts and Insertion Moves**.
    # The key difference from the current program is the **Insertion Move** 
    # and **Random Restarts**.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Start with seed
    current_net = _build_seed()
    if not _verify_network(current_net):
        return current_net
        
    best_net = current_net[:]
    best_size = len(current_net)
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Strategy: Iterative Local Search with "Escape" moves
    # 1. Try to remove a comparator.
    # 2. If stuck, try to ADD a random comparator (temporarily increasing size) 
    #    to break symmetry or enable new removals.
    # 3. If size improves, update best.
    # 4. If stuck for too long, restart from best_net with a different random seed.
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        improved = False
        stuck_counter = 0
        
        # Try to remove comparators until no more can be removed
        # We randomize the order of attempts to avoid bias
        indices = list(range(len(current_net)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            if idx >= len(current_net):
                continue
                
            removed_comp = current_net[idx]
            new_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network(new_net):
                current_net = new_net
                improved = True
                if len(current_net) < best_size:
                    best_net = current_net[:]
                    best_size = len(current_net)
                # After removal, the indices shift. 
                # We break and restart the shuffle to be safe and simple.
                # This is slightly less efficient but safer.
                break
        
        if not improved:
            # We are in a local minimum for removals.
            # Try an "Escape" move: Add a random comparator.
            # This increases size, but might allow removing 2+ others later.
            # We only do this if we have time.
            
            # Select a random comparator NOT in current_net
            # To speed up, we can just pick a random pair and check if it's already there.
            # Or maintain a set of current comps.
            comp_set = set(current_net)
            
            # Try a few random additions
            found_addition = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                c = rng.choice(all_comps)
                if c not in comp_set:
                    new_net = current_net + [c]
                    if _verify_network(new_net):
                        current_net = new_net
                        found_addition = True
                        break
            
            if found_addition:
                # Reset stuck counter, continue loop to try removals again
                continue
            else:
                # Couldn't even find a valid addition? 
                # Or we tried and failed.
                # If we are truly stuck, restart from best_net
                # with a different randomization.
                
                # Check if we should restart
                # Simple heuristic: if we've been stuck for a while, restart.
                # Since we don't track global time in this sub-loop easily, 
                # let's just restart from best_net immediately if we can't improve.
                # This is a "Random Restart" strategy.
                
                if len(current_net) == best_size:
                    # We are at the best size found so far, but stuck.
                    # Restart from best_net with a fresh shuffle.
                    current_net = best_net[:]
                    # Shuffle the network order? 
                    # The order matters for the "removal" greedy pass.
                    rng.shuffle(current_net)
                else:
                    # We are larger than best? This shouldn't happen if we always 
                    # revert failed removals and only keep valid additions temporarily.
                    # Wait, we keep valid additions. So current_net might be larger than best_net.
                    # If we can't remove anything, and we added something, 
                    # we are worse than best.
                    # Revert to best_net.
                    current_net = best_net[:]
                    rng.shuffle(current_net)
                    
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
