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

def _greedy_color_upper_bound_fast(conflict_bits, candidates_mask):
    """
    Fast greedy coloring upper bound using bit operations.
    Returns upper bound on max independent set size.
    """
    if candidates_mask == 0:
        return 0
    
    remaining = candidates_mask
    num_colors = 0
    
    while remaining:
        num_colors += 1
        # Greedily pick a maximal independent set from remaining
        # Process nodes in order of increasing degree within remaining
        uncolored = remaining
        current_color_set = 0
        
        # Build list of (degree, node) for nodes in uncolored
        temp = uncolored
        nodes_list = []
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = (conflict_bits[v] & uncolored).bit_count()
            nodes_list.append((deg, v))
        nodes_list.sort()
        
        for deg, v in nodes_list:
            if not (uncolored & (1 << v)):
                continue
            if not (conflict_bits[v] & current_color_set):
                current_color_set |= (1 << v)
                uncolored &= ~(1 << v)
        
        remaining = uncolored
    
    return num_colors

def _max_independent_set_bnb(conflict_bits, initial_set, candidates_mask, start_time, deadline):
    """
    Branch and bound for maximum independent set with improved heuristics.
    initial_set: set of already-selected nodes
    candidates_mask: bitmask of nodes that can still be added (no conflicts with initial_set)
    Returns the best independent set found.
    """
    best_set = set(initial_set)
    best_size = len(initial_set)
    
    def bnb(cands_mask, current_size):
        nonlocal best_set, best_size
        
        if (time.time() - start_time) > deadline:
            return
        
        if cands_mask == 0:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set from cands_mask... we need to track the actual set
                # Actually let's use a different approach: track set explicitly
            return
        
        # Quick upper bound: if even taking all candidates doesn't beat best, prune
        cands_count = cands_mask.bit_count()
        if current_size + cands_count <= best_size:
            return
        
        # Upper bound: current size + greedy coloring of remaining candidates
        ub = current_size + _greedy_color_upper_bound_fast(conflict_bits, cands_mask)
        if ub <= best_size:
            return
        
        # Select a node to branch on: pick the one with highest degree in cands
        node = None
        max_score = -1
        temp_mask = cands_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            deg = (conflict_bits[v] & cands_mask).bit_count()
            if deg > max_score:
                max_score = deg
                node = v
        
        if node is None:
            return
        
        # Branch 1: include node
        new_cands = cands_mask & ~(1 << node) & ~conflict_bits[node]
        bnb(new_cands, current_size + 1)
        
        # Branch 2: exclude node
        new_cands2 = cands_mask & ~(1 << node)
        bnb(new_cands2, current_size)
    
    # We need to track the actual set, so let's use a different approach
    # Store the best mask
    best_mask = 0
    for x in initial_set:
        best_mask |= (1 << x)
    
    def bnb2(cands_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if (time.time() - start_time) > deadline:
            return
        
        if cands_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        cands_count = cands_mask.bit_count()
        if current_size + cands_count <= best_size:
            return
        
        ub = current_size + _greedy_color_upper_bound_fast(conflict_bits, cands_mask)
        if ub <= best_size:
            return
        
        node = None
        max_score = -1
        temp_mask = cands_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            deg = (conflict_bits[v] & cands_mask).bit_count()
            if deg > max_score:
                max_score = deg
                node = v
        
        if node is None:
            return
        
        # Branch 1: include node
        new_cands = cands_mask & ~(1 << node) & ~conflict_bits[node]
        new_mask = current_mask | (1 << node)
        bnb2(new_cands, new_mask, current_size + 1)
        
        # Branch 2: exclude node
        new_cands2 = cands_mask & ~(1 << node)
        bnb2(new_cands2, current_mask, current_size)
    
    init_size = len(initial_set)
    bnb2(candidates_mask, best_mask, init_size)
    
    # Reconstruct set from best_mask
    result = set()
    m = best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        result.add(v)
    return result

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
    """Greedily add compatible elements to set S using bitset operations."""
    S = set(S)
    
    # Build a bitmask of current S
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    # Find all candidates that don't conflict with S
    # A candidate x is compatible if conflict_bits[x] & S_mask == 0
    # But we also need to ensure pairwise compatibility among added elements
    
    # Use a greedy approach: repeatedly add the element with fewest conflicts
    remaining = 0
    for x in range(1, N + 1):
        if not (S_mask & (1 << x)):
            if not (conflict_bits[x] & S_mask):
                remaining |= (1 << x)
    
    # Now greedily add from remaining, preferring elements with fewer conflicts among remaining
    while remaining:
        if (time.time() - start_time) > deadline:
            break
        
        # Find element in remaining with fewest conflicts within remaining
        best_x = None
        min_conf = float('inf')
        temp = remaining
        while temp:
            lsb = temp & (-temp)
            x = lsb.bit_length() - 1
            temp ^= lsb
            conf = bin(conflict_bits[x] & remaining).count('1')
            if conf < min_conf:
                min_conf = conf
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x to S
        S.add(best_x)
        S_mask |= (1 << best_x)
        # Remove best_x and all its conflicts from remaining
        remaining &= ~(1 << best_x)
        remaining &= ~conflict_bits[best_x]
    
    return S

def _local_search_fast(conflict_bits, S, start_time, deadline):
    """Fast local search: try to add elements that become compatible after removing one element."""
    S = set(S)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    improved = True
    iterations = 0
    
    while improved and (time.time() - start_time) < deadline and iterations < 50:
        improved = False
        iterations += 1
        
        # Try 1-swap: remove one element, try to add one or more new elements
        for elem in list(S):
            if (time.time() - start_time) > deadline:
                return S
            
            S_without_mask = S_mask & ~(1 << elem)
            
            # Find all elements not in S that are compatible with S_without
            addable_mask = 0
            for x in range(1, N + 1):
                if x == elem:
                    continue
                if S_mask & (1 << x):
                    continue
                if not (conflict_bits[x] & S_without_mask):
                    addable_mask |= (1 << x)
            
            if addable_mask == 0:
                continue
            
            # Greedily fill from addable_mask
            new_S_mask = S_without_mask
            remaining_add = addable_mask
            
            while remaining_add:
                if (time.time() - start_time) > deadline:
                    break
                
                # Pick element with fewest conflicts in remaining_add
                best_x = None
                min_conf = float('inf')
                temp = remaining_add
                while temp:
                    lsb = temp & (-temp)
                    x = lsb.bit_length() - 1
                    temp ^= lsb
                    conf = (conflict_bits[x] & remaining_add).bit_count()
                    if conf < min_conf:
                        min_conf = conf
                        best_x = x
                
                if best_x is None:
                    break
                
                new_S_mask |= (1 << best_x)
                remaining_add &= ~(1 << best_x)
                remaining_add &= ~conflict_bits[best_x]
            
            new_size = new_S_mask.bit_count()
            if new_size > S_mask.bit_count():
                # Reconstruct S from new_S_mask
                S = set()
                m = new_S_mask
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    S.add(v)
                S_mask = new_S_mask
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
    high_composite_numbers = [360, 240, 180, 120, 480, 336, 420, 504, 288, 192, 168, 144, 96, 72, 48, 36, 24, 12, 6, 4, 2, 1]
    
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
    
    # Strategy 2: Local search first to get a better starting point for B&B
    if (time.time() - start_time) < deadline - 10:
        S_ls = _local_search_fast(conflict_bits, best_S, start_time, deadline - 10)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = S_ls
    
    # Strategy 3: Branch and bound to improve
    if (time.time() - start_time) < deadline - 5:
        # Compute which nodes can be added to best_S (no conflicts)
        best_S_mask = 0
        for x in best_S:
            best_S_mask |= (1 << x)
        
        candidates_mask = 0
        for x in range(1, N + 1):
            if not (best_S_mask & (1 << x)):
                if not (conflict_bits[x] & best_S_mask):
                    candidates_mask |= (1 << x)
        
        if candidates_mask:
            # Run B&B
            result = _max_independent_set_bnb(conflict_bits, best_S, candidates_mask, start_time, deadline - 5)
            if len(result) > best_size:
                best_size = len(result)
                best_S = result
    
    # Strategy 4: Local search after B&B
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
