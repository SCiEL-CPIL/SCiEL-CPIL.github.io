"""Branch-and-bound maximum independent set for equal-LCM triple avoidance."""
import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
    
    # conflict[node][other] = bitmask of partners p such that (node, other, p) is a bad triple
    conflict = [dict() for _ in range(n)]
    for a, b, c in bad_triples:
        conflict[a][b] = conflict[a].get(b, 0) | (1 << c)
        conflict[b][a] = conflict[b].get(a, 0) | (1 << c)
        conflict[c][a] = conflict[c].get(a, 0) | (1 << b)
        conflict[c][b] = conflict[c].get(b, 0) | (1 << a)
    
    # For fast _can_add: precompute for each node a list of (other, partners_mask)
    conflict_lists = []
    for i in range(n):
        conflict_lists.append(list(conflict[i].items()))
    
    # Precompute bit_count cache for common operations
    all_mask = (1 << n) - 1
    
    best_size = 0
    best_solution_mask = 0
    
    def _can_add(selected_mask, node):
        """Check if adding node to selected_mask doesn't complete any bad triple."""
        for other, partners in conflict_lists[node]:
            if (selected_mask >> other) & 1:
                if selected_mask & partners:
                    return False
        return True
    
    def _matching_bound(available_mask):
        """
        Upper bound via greedy matching on the hypergraph induced by available_mask.
        Each hyperedge (bad triple) can cover at most 2 of its vertices in an independent set.
        Greedily match hyperedges to get a lower bound on the number of "excluded" vertices,
        giving an upper bound on the independent set size.
        """
        if not available_mask:
            return 0
        
        # Greedy matching: repeatedly find a bad triple with all 3 vertices available,
        # mark 2 of them as "covered", continue.
        # The number of available vertices not covered gives an upper bound.
        # Actually: each matched hyperedge means at least 1 vertex must be excluded.
        # So upper_bound = popcount(available) - num_matched_edges (roughly)
        # Better: use a proper matching bound.
        
        # Simple bound: for each bad triple fully in available, at least 1 must be excluded.
        # Use a greedy set cover / matching approach.
        # For speed, use a simple bound: count how many disjoint bad triples exist.
        
        # Even simpler and effective: the trivial bound is popcount(available).
        # A better bound: find a maximal matching of hyperedges.
        # Each hyperedge in the matching forces at least 1 exclusion.
        
        # For performance, use a lightweight version:
        # Count vertices that participate in any bad triple within available.
        # Those vertices can be at most (count - 1) if they form a connected component... 
        # This is getting complex. Use the simple bound for now but add a quick improvement:
        
        pc = available_mask.bit_count()
        
        # Quick improvement: find if there's a bad triple fully in available.
        # If so, at least 1 must be excluded from that triple.
        # Do a few rounds of this.
        extra = 0
        am = available_mask
        # Limit iterations for speed
        for _ in range(20):
            # Find first bad triple with all 3 in am
            found = False
            for a, b, c in bad_triples:
                if (am & (1 << a)) and (am & (1 << b)) and (am & (1 << c)):
                    extra += 1
                    # Remove one vertex from am to continue matching
                    am &= ~(1 << a)
                    found = True
                    break
            if not found:
                break
        
        return pc - extra
    
    def _bbs(idx, sel_mask, rem_mask, node_order, time_left):
        """Branch and bound search with pre-sorted node order."""
        nonlocal best_size, best_solution_mask
        
        if time.time() > time_left:
            return
        
        cur_size = sel_mask.bit_count()
        
        # Upper bound: current + matching bound on remaining
        ub = cur_size + _matching_bound(rem_mask)
        if ub <= best_size:
            return
        
        if idx >= len(node_order):
            if cur_size > best_size:
                best_size = cur_size
                best_solution_mask = sel_mask
            return
        
        node = node_order[idx]
        bit = 1 << node
        
        # Option 1: Include node
        if _can_add(sel_mask, node):
            new_sel = sel_mask | bit
            new_rem = rem_mask & ~bit
            _bbs(idx + 1, new_sel, new_rem, node_order, time_left)
        
        # Option 2: Exclude node
        _bbs(idx + 1, sel_mask, rem_mask & ~bit, node_order, time_left)
    
    # Initial greedy to get a good starting lower bound
    def _greedy_init(order):
        selected = 0
        for node in order:
            if time.time() > deadline - 10:
                break
            if _can_add(selected, node):
                selected |= (1 << node)
        return selected
    
    # Try multiple orderings for initial greedy
    conflict_counts = [len(conflict_lists[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda i: conflict_counts[i])
    order2 = list(reversed(order1))
    order3 = list(range(n))
    order4 = list(reversed(range(n)))
    
    best_init = 0
    for order in [order1, order2, order3, order4]:
        sol = _greedy_init(order)
        if sol.bit_count() > best_init.bit_count():
            best_init = sol
    
    best_size = best_init.bit_count()
    best_solution_mask = best_init
    
    # Now run branch and bound from the greedy solution
    remaining = all_mask & ~best_init
    
    # Sort remaining nodes by conflict count (ascending) for better pruning
    rem_list = []
    r = remaining
    while r:
        low_bit = r & (-r)
        node = low_bit.bit_length() - 1
        r ^= low_bit
        rem_list.append(node)
    
    # Sort by number of conflicts (ascending)
    rem_list.sort(key=lambda x: len(conflict_lists[x]))
    
    # Run B&B
    _bbs(0, best_solution_mask, remaining, rem_list, deadline - 5)
    
    # Final check: try to add any remaining nodes to the best solution
    if time.time() < deadline - 2:
        for node in range(n):
            if not (best_solution_mask >> node) & 1:
                if _can_add(best_solution_mask, node):
                    best_solution_mask |= (1 << node)
    
    # Convert mask to list
    result = [i + 1 for i in range(n) if (best_solution_mask >> i) & 1]
    
    if not result:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
