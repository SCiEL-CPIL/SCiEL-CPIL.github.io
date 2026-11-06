"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    try:
        from autoresearch_suite.science_tcs_expansion import xor_baseline
    except ImportError:
        # Fallback if import fails, though it should not
        def xor_baseline(m, d):
            return {'gates': [], 'outputs': []}

    m, n = matrix.shape
    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Initial state
    # Wires: 0..n-1 are inputs, n is zero
    initial_available = set([1 << i for i in range(n)] + [0])
    initial_targets = set(rows)
    
    # Wire map: bitmask -> wire_index
    wire_map = {}
    for i in range(n):
        wire_map[1 << i] = i
    wire_map[0] = n
    
    # Beam Search
    beam_width = 20
    max_depth = depth_limit
    
    # State: (available, targets, gates, wire_map)
    current_beam = [ (initial_available, initial_targets, [], wire_map) ]
    
    best_solution = None
    best_cost = float('inf')
    
    # If initial state is already a solution (all targets in available)
    if not initial_targets:
        best_solution = ([], wire_map)
        best_cost = 0

    for depth in range(max_depth):
        if time.time() > deadline - 5.0:
            break
            
        next_beam_candidates = []
        
        for state in current_beam:
            available, targets, gates, wmap = state
            
            # If all targets met, we have a solution
            if not targets:
                cost = len(gates)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (gates, wmap)
                # We can stop expanding this state
                continue
            
            # Generate candidates
            avail_list = list(available)
            L = len(avail_list)
            
            # Optimization: skip if too many candidates
            # But L is small initially.
            
            for i in range(L):
                a = avail_list[i]
                for j in range(i+1, L):
                    b = avail_list[j]
                    u = a ^ b
                    if u in available:
                        continue
                    
                    # New state
                    new_available = available | {u}
                    new_targets = targets.copy()
                    
                    # Check if u completes any target
                    if u in new_targets:
                        new_targets.remove(u)
                    
                    # Heuristic:
                    # 1. Number of targets met
                    # 2. Number of targets that are "1-step away"
                    #    t is 1-step away if t ^ x in new_available for some x
                    #    This is expensive. Let's use a simpler proxy.
                    #    Proxy: -len(new_targets)
                    
                    # To make it stronger, let's add a small bonus for each target
                    # that is now closer.
                    # We can approximate closeness by checking if t ^ a in new_available
                    # for the specific a,b that formed u.
                    # If t ^ u in available, then t is 1-step away from u.
                    # If t ^ a in available, t is 1-step away from a.
                    
                    # Let's just use the number of targets met as primary progress.
                    
                    new_gates = gates + [(wmap[a], wmap[b])]
                    new_wmap = wmap.copy()
                    new_wmap[u] = n + 1 + len(gates)
                    
                    # Score for sorting:
                    # Primary: minimize gates (len(new_gates))
                    # Secondary: maximize progress (-len(new_targets))
                    # Tertiary: random tie-breaker to avoid bias?
                    
                    # We store: (gate_count, -progress, available, targets, gates, wmap)
                    progress = len(targets) - len(new_targets)
                    next_beam_candidates.append( (len(new_gates), -progress, new_available, new_targets, new_gates, new_wmap) )
            
            # Prune next_beam_candidates
            if not next_beam_candidates:
                break
            
            # Sort by (gate_count, -progress)
            next_beam_candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Keep top beam_width unique states
            current_beam = []
            seen_states = set()
            for c in next_beam_candidates:
                key = (frozenset(c[2]), frozenset(c[3]))
                if key in seen_states:
                    continue
                seen_states.add(key)
                current_beam.append( (c[2], c[3], c[4], c[5]) )
                if len(current_beam) >= beam_width:
                    break
            
            if not current_beam:
                break
            
            # Check if any state in current_beam is a solution
            for state in current_beam:
                available, targets, gates, wmap = state
                if not targets:
                    cost = len(gates)
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = (gates, wmap)

    # Construct output
    if best_solution:
        gates, wmap = best_solution
        outputs = []
        for r in rows:
            outputs.append(wmap[r])
        return {'gates': gates, 'outputs': outputs}
    else:
        # Fallback to baseline
        return xor_baseline(matrix, depth_limit)

# EVOLVE-BLOCK-END
