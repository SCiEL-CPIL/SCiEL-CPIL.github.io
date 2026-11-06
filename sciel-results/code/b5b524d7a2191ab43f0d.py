"""Size-efficient sorting network search for N=16 wires.

Hybrid Evolutionary Local Search approach:
- Uses fast bit-parallel verification (zero-one principle).
- Maintains a population of valid, minimized sorting networks.
- Uses mutation (insert random comparator) and crossover (splice) to generate offspring.
- Each offspring is strictly minimized via iterative greedy removal.
- Selection preserves the smallest valid networks found.
- This combines the structural exploration of GP with the rigorous refinement of local search.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def minimize_network(net, deadline):
    """
    Iteratively remove redundant comparators.
    Returns the minimized network.
    """
    if not net:
        return net
    
    # First pass: remove obvious no-ops quickly
    # We do a full scan and remove any comparator that is a no-op.
    # Then we repeat until no changes are made.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards to allow index stability if we were to delete in place, 
        # but we build a new list.
        new_net = []
        for i, comp in enumerate(net):
            # Check if removing this comparator keeps the network valid
            # Optimization: Only check if the comparator is *potentially* redundant.
            # A comparator (i, j) is redundant if wire i <= wire j for all inputs 
            # *before* this comparator is applied.
            
            # To check this efficiently without full simulation of the whole network 
            # for every removal attempt (which is O(N^2 * N_inputs)), we can 
            # just try the removal and verify the whole network. 
            # Given N=16 and ~60 comparators, and fast_verify being very fast,
            # this is acceptable.
            
            if time.time() > deadline:
                break
                
            candidate = net[:i] + net[i+1:]
            if fast_verify(candidate):
                # Removal is valid
                # We found a redundant comparator. 
                # To be aggressive, we remove it and restart the scan.
                net = candidate
                changed = True
                break # Restart the while loop to rescan the new network
            else:
                new_net.append(comp)
        
        if not changed:
            # If we went through the whole list without finding a removable comparator
            # in this pass, the network is locally minimal.
            # However, the 'break' logic above means we only continue if we found one.
            # If 'changed' is False, we exit the while loop.
            pass
            
    return net

def construct():
    deadline = time.time() + 85  # 5s margin for overhead
    
    random.seed(42)
    
    # Initialize population with the Batcher sorter, minimized
    initial_net = bitonic_sorter(N)
    initial_net = minimize_network(initial_net, deadline)
    
    # Population: list of (size, network)
    # We want to keep distinct networks. 
    # Using a set of tuples of tuples is memory heavy but safe for ~60 comps.
    # Alternatively, just keep the best K and hope for diversity, or use a simple hash.
    
    best_net = initial_net
    best_size = len(best_net)
    
    # Population of top candidates
    # Limit population size to keep memory and time under control
    MAX_POP = 20
    population = [(best_size, tuple(best_net))]
    
    # If initial is already 60, we are done. But let's search just in case or to find 59 (impossible but check).
    if best_size == 60:
         # We can still try to find 59, but it's known to be impossible.
         # We can return early to save time, but let's run a bit to be sure.
         pass

    start_time = time.time()
    
    while time.time() < deadline:
        # Generate offspring
        
        # Strategy 1: Mutation (Insert + Minimize)
        if len(population) > 0:
            parent_size, parent_net_tuple = random.choice(population)
            parent_net = list(parent_net_tuple)
            
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(parent_net))
            
            mutated_net = parent_net[:pos] + [(i, j)] + parent_net[pos:]
            
            # Minimize
            # Optimization: If the inserted comparator is immediately redundant, 
            # minimize will remove it quickly.
            minimized_mut = minimize_network(mutated_net, deadline)
            
            if minimized_mut:
                mut_size = len(minimized_mut)
                mut_tuple = tuple(minimized_mut)
                
                # Check if it's new and better
                if mut_size < best_size:
                    best_net = minimized_mut
                    best_size = mut_size
                    population.append((mut_size, mut_tuple))
                    population.sort(key=lambda x: x[0])
                    if len(population) > MAX_POP:
                        population = population[:MAX_POP]
                elif mut_size == best_size:
                    # Check if distinct
                    if mut_tuple not in [p[1] for p in population]:
                        population.append((mut_size, mut_tuple))
                        population.sort(key=lambda x: x[0])
                        if len(population) > MAX_POP:
                            population = population[:MAX_POP]

        # Strategy 2: Crossover (Splice + Minimize)
        if len(population) >= 2:
            # Pick two distinct parents
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            if p1_idx == p2_idx:
                continue
                
            _, p1_tuple = population[p1_idx]
            _, p2_tuple = population[p2_idx]
            
            p1 = list(p1_tuple)
            p2 = list(p2_tuple)
            
            # Splice: Take first half of P1 and second half of P2
            # Lengths might differ. 
            # We splice at a position relative to the shorter one or random.
            cross_pos = random.randint(0, min(len(p1), len(p2)))
            
            crossed_net = p1[:cross_pos] + p2[cross_pos:]
            
            # Note: The crossed network might be invalid.
            # Minimize will only work if it's valid or can be made valid by removal?
            # NO. Minimize only removes. It does not add. 
            # So if the crossover produces an invalid network, minimize_network will 
            # return the same invalid network (or a reduced invalid one).
            # We MUST verify validity before accepting.
            
            # Actually, minimize_network calls fast_verify. 
            # If the input to minimize is invalid, fast_verify returns False.
            # The current minimize_network logic:
            #   candidate = net[:i] + net[i+1:]
            #   if fast_verify(candidate): ...
            # If the original net is invalid, removing a comparator might make it valid?
            # Yes, if the invalid comparator was causing the issue. 
            # But typically, invalid networks need *additions* to be fixed.
            # So crossover is risky. 
            # We should verify the crossed_net first. If invalid, discard.
            
            if fast_verify(crossed_net):
                minimized_cross = minimize_network(crossed_net, deadline)
                cross_size = len(minimized_cross)
                cross_tuple = tuple(minimized_cross)
                
                if cross_size < best_size:
                    best_net = minimized_cross
                    best_size = cross_size
                    population.append((cross_size, cross_tuple))
                    population.sort(key=lambda x: x[0])
                    if len(population) > MAX_POP:
                        population = population[:MAX_POP]
                elif cross_size == best_size:
                    if cross_tuple not in [p[1] for p in population]:
                        population.append((cross_size, cross_tuple))
                        population.sort(key=lambda x: x[0])
                        if len(population) > MAX_POP:
                            population = population[:MAX_POP]

        # Strategy 3: Random Restart (Occasionally)
        # If we stagnate, try a new random seed or a different initial structure?
        # We don't have other known structures. 
        # We can try to "perturb" the best net heavily.
        
        if time.time() > deadline:
            break

    return best_net

