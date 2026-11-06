"""Branch-and-bound maximum independent set for equal-LCM triple avoidance."""
import time
import itertools
from math import gcd
from functools import reduce

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
    
    # For each node, precompute: for each other node, the set of partners that form a bad triple
    # conflict[node][other] = set of partners p such that (node, other, p) is a bad triple
    # We use bitmasks for fast intersection
    conflict = [dict() for _ in range(n)]
    for a, b, c in bad_triples:
        conflict[a].setdefault(b, 0)
        conflict[a][b] |= (1 << c)
        conflict[b].setdefault(a, 0)
        conflict[b][a] |= (1 << c)
        conflict[c].setdefault(a, 0)
        conflict[c][a] |= (1 << b)
        conflict[c].setdefault(b, 0)
        conflict[c][b] |= (1 << a)
    
    # Convert to sorted lists for iteration
    conflict_lists = []
    for i in range(n):
        conflict_lists.append(list(conflict[i].items()))
    
    # Bitmask for all nodes
    all_mask = (1 << n) - 1
    
    best_size = 0
    best_solution_mask = 0
    
    def _can_add(selected_mask, node):
        """Check if adding node to selected_mask doesn't complete any bad triple."""
        for other, partners in conflict_lists[node]:
            if (selected_mask >> other) & 1:
                # Check if any partner is in selected_mask
                if selected_mask & partners:
                    return False
        return True
    
    def _upper_bound(selected_mask, remaining_mask):
        """Upper bound: current size + popcount of remaining."""
        return bin(selected_mask).count('1') + bin(remaining_mask).count('1')
    
    def _branch_and_bound(selected_mask, remaining_mask, depth, time_left):
        """Branch and bound search."""
        nonlocal best_size, best_solution_mask
        
        if time.time() > time_left:
            return
        
        current_size = bin(selected_mask).count('1')
        
        # Prune: if current + remaining can't beat best
        if current_size + bin(remaining_mask).count('1') <= best_size:
            return
        
        # Try to add as many as possible greedily to get a good lower bound
        # First, find a feasible greedy completion
        greedy_mask = selected_mask
        rem = remaining_mask
        while rem:
            # Find a node that can be added
            added = False
            # Iterate through remaining nodes
            r = rem
            while r:
                low_bit = r & (-r)
                node = low_bit.bit_length() - 1
                r ^= low_bit
                if _can_add(greedy_mask, node):
                    greedy_mask |= (1 << node)
                    rem &= ~(1 << node)
                    added = True
                    break
            if not added:
                break
        
        greedy_size = bin(greedy_mask).count('1')
        if greedy_size > best_size:
            best_size = greedy_size
            best_solution_mask = greedy_mask
        
        # Now do branch and bound on the remaining nodes
        # Sort remaining nodes by conflict count (ascending) for better pruning
        rem_list = []
        r = remaining_mask
        while r:
            low_bit = r & (-r)
            node = low_bit.bit_length() - 1
            r ^= low_bit
            rem_list.append(node)
        
        # Sort by number of conflicts (ascending) - nodes with fewer conflicts are easier to include
        rem_list.sort(key=lambda x: len(conflict_lists[x]))
        
        def _bbs(idx, sel_mask, rem_mask):
            nonlocal best_size, best_solution_mask
            
            if time.time() > time_left:
                return
            
            cur_size = bin(sel_mask).count('1')
            
            # Upper bound
            if cur_size + bin(rem_mask).count('1') <= best_size:
                return
            
            if idx >= len(rem_list):
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution_mask = sel_mask
                return
            
            node = rem_list[idx]
            
            # Option 1: Include node
            if _can_add(sel_mask, node):
                new_sel = sel_mask | (1 << node)
                new_rem = rem_mask & ~(1 << node)
                # Remove nodes that conflict with this addition? No, we check dynamically.
                _bbs(idx + 1, new_sel, new_rem)
            
            # Option 2: Exclude node
            _bbs(idx + 1, sel_mask, rem_mask & ~(1 << node))
        
        _bbs(0, selected_mask, remaining_mask)
    
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
        if bin(sol).count('1') > bin(best_init).count('1'):
            best_init = sol
    
    best_size = bin(best_init).count('1')
    best_solution_mask = best_init
    
    # Now run branch and bound from the greedy solution
    # Remaining nodes are those not in best_init
    remaining = all_mask & ~best_init
    
    # We can also try B&B from scratch with a tighter bound
    # But starting from a good greedy solution is usually better
    
    # Run B&B
    _branch_and_bound(best_solution_mask, remaining, 0, deadline - 5)
    
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
