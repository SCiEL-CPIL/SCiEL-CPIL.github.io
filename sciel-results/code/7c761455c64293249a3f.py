"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _build_conflict_bitsets():
    """Build conflict graph as bitsets (Python ints) for fast operations."""
    # conflict_bits[i] is a bitmask where bit j is set if lcm(i,j) > N
    conflict_bits = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if _lcm(a, b) > N:
                conflict_bits[a] |= (1 << b)
                conflict_bits[b] |= (1 << a)
    return conflict_bits

def _greedy_color_upper_bound(conflict_bits, candidates_mask):
    """
    Compute an upper bound on the maximum independent set size using greedy coloring.
    candidates_mask: bitmask of candidate nodes
    """
    if candidates_mask == 0:
        return 0
    
    color = {}
    # Process nodes in order of decreasing degree within candidates
    nodes = []
    for v in range(1, N + 1):
        if (candidates_mask >> v) & 1:
            deg = bin(conflict_bits[v] & candidates_mask).count('1')
            nodes.append((deg, v))
    nodes.sort(reverse=True)
    
    for deg, node in nodes:
        used_colors = set()
        # Check neighbors in candidates that already have colors
        neighbor_mask = conflict_bits[node] & candidates_mask
        while neighbor_mask:
            lsb = neighbor_mask & (-neighbor_mask)
            nb = lsb.bit_length() - 1
            neighbor_mask ^= lsb
            if nb in color:
                used_colors.add(color[nb])
        c = 1
        while c in used_colors:
            c += 1
        color[node] = c
    
    return max(color.values()) if color else 0

def _max_independent_set_bnb(conflict_bits, initial_set, candidates_mask, start_time, deadline):
    """
    Branch and bound for maximum independent set.
    initial_set: set of already-selected nodes
    candidates_mask: bitmask of nodes that can still be added (no conflicts with initial_set)
    Returns the best independent set found.
    """
    best_set = set(initial_set)
    best_size = len(initial_set)
    
    def bnb(cands_mask, current_set):
        nonlocal best_set, best_size
        
        if (time.time() - start_time) > deadline:
            return
        
        if cands_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: current size + greedy coloring of remaining candidates
        ub = len(current_set) + _greedy_color_upper_bound(conflict_bits, cands_mask)
        if ub <= best_size:
            return
        
        # Select a node to branch on: pick the one with highest degree in cands
        node = None
        max_deg = -1
        temp_mask = cands_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            deg = bin(conflict_bits[v] & cands_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                node = v
        
        if node is None:
            return
        
        # Branch 1: include node
        new_cands = cands_mask & ~(1 << node) & ~conflict_bits[node]
        new_current = current_set | {node}
        bnb(new_cands, new_current)
        
        # Branch 2: exclude node
        new_cands2 = cands_mask & ~(1 << node)
        bnb(new_cands2, current_set)
    
    bnb(candidates_mask, initial_set)
    return best_set

def _get_divisors(m):
    """Get all divisors of m that are <= N."""
    divs = set()
    for i in range(1, int(m**0.5) + 1):
        if m % i == 0:
            if i <= N:
                divs.add(i)
            if m // i <= N:
                divs.add(m // i)
    return divs

def _greedy_fill(conflict_bits, S, start_time, deadline):
    """Greedily add compatible elements to set S."""
    S = set(S)
    # Sort candidates by number of conflicts (ascending) - prefer less conflicting
    candidates = []
    for x in range(1, N + 1):
        if x not in S:
            # Count conflicts with current S
            conf_count = 0
            for y in S:
                if (conflict_bits[x] >> y) & 1:
                    conf_count += 1
            candidates.append((conf_count, x))
    
    candidates.sort()
    
    for conf_count, x in candidates:
        if (time.time() - start_time) > deadline:
            break
        if x in S:
            continue
        # Check if x conflicts with any element in S
        compatible = True
        for y in S:
            if (conflict_bits[x] >> y) & 1:
                compatible = False
                break
        if compatible:
            S.add(x)
    
    return S

def _local_search_fast(conflict_bits, S, start_time, deadline):
    """Fast local search using bitsets: try to swap elements to grow the set."""
    S = set(S)
    improved = True
    iterations = 0
    
    while improved and (time.time() - start_time) < deadline and iterations < 100:
        improved = False
        iterations += 1
        current_size = len(S)
        S_list = sorted(S)
        
        for elem in S_list:
            if (time.time() - start_time) > deadline:
                return S
            
            S_without = S - {elem}
            
            # Find best replacement: an element not in S that doesn't conflict with S_without
            # and allows adding more elements
            best_new_elem = None
            best_gain = 0
            
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Check if x is compatible with S_without
                compatible = True
                for y in S_without:
                    if (conflict_bits[x] >> y) & 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # If we replace elem with x, can we add more?
                # For now, just check if this swap maintains or increases size
                # A simple heuristic: prefer x that conflicts with fewer other candidates
                new_S = S_without | {x}
                if len(new_S) > current_size:
                    # This shouldn't happen in a swap, but just in case
                    best_new_elem = x
                    best_gain = 1
                    break
                elif len(new_S) == current_size:
                    # Count how many additional elements could be added
                    addable_count = 0
                    for z in range(1, N + 1):
                        if z in new_S:
                            continue
                        compat = True
                        for y in new_S:
                            if (conflict_bits[z] >> y) & 1:
                                compat = False
                                break
                        if compat:
                            addable_count += 1
                    if addable_count > best_gain:
                        best_gain = addable_count
                        best_new_elem = x
            
            if best_new_elem is not None and best_gain > 0:
                S = (S - {elem}) | {best_new_elem}
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Build conflict graph as bitsets
    conflict_bits = _build_conflict_bitsets()
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Try divisor sets of highly composite numbers
    # These give good initial lower bounds
    high_composite_numbers = [360, 240, 180, 120, 480, 336, 420, 504, 360, 288, 240, 192, 168, 144, 120, 96, 72, 48, 36, 24, 12, 6, 4, 2, 1]
    # Add some more
    high_composite_numbers += [420, 480, 504, 336, 360, 288, 240, 192, 168, 144, 120, 96, 72, 48, 36, 24, 12, 6, 4, 2]
    
    seen_divisor_sets = set()
    for m in high_composite_numbers:
        if (time.time() - start_time) > deadline:
            break
        divs = _get_divisors(m)
        # Use a hash of the divisor set to avoid duplicates
        divs_frozen = frozenset(divs)
        if divs_frozen in seen_divisor_sets:
            continue
        seen_divisor_sets.add(divs_frozen)
        
        # Greedy fill
        S = _greedy_fill(conflict_bits, divs, start_time, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Ensure we have at least {1}
    if not best_S:
        best_S = {1}
        best_size = 1
    
    # Strategy 2: Branch and bound to improve
    if (time.time() - start_time) < deadline - 5:
        # Compute which nodes can be added to best_S (no conflicts)
        candidates_mask = 0
        for x in range(1, N + 1):
            if x not in best_S:
                # Check if x conflicts with any element in best_S
                conflicts = False
                for y in best_S:
                    if (conflict_bits[x] >> y) & 1:
                        conflicts = True
                        break
                if not conflicts:
                    candidates_mask |= (1 << x)
        
        if candidates_mask:
            # Run B&B
            result = _max_independent_set_bnb(conflict_bits, best_S, candidates_mask, start_time, deadline - 5)
            if len(result) > best_size:
                best_size = len(result)
                best_S = result
    
    # Strategy 3: Local search
    if (time.time() - start_time) < deadline - 2:
        S_final = _local_search_fast(conflict_bits, best_S, start_time, deadline - 2)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    # Final greedy fill
    if (time.time() - start_time) < deadline - 1:
        S_final = _greedy_fill(conflict_bits, best_S, start_time, deadline - 1)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    if not best_S:
        best_S = {1}
    
    return list(best_S)
# EVOLVE-BLOCK-END
