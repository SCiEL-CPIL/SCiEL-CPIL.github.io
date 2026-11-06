"""Optimized program for finding max subset of [1,128] avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct numbers have all pairwise LCMs equal."""
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if time.time() > deadline - 30:
            break
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements
    
    # Build bitset-based conflict structures
    # For each node, store bitsets of pairs that form bad triples with it
    # pair_bits[i] = set of bit masks where each bit represents a pair (j,k) 
    # Actually, let's use a different approach: for each node i, store a list of (j,k) pairs
    # And for fast checking, use a 2D array or bitset approach
    
    # For each node, store the set of other nodes it conflicts with in triples
    # conflict[i] = set of j such that there exists k where (i,j,k) is a bad triple
    # But we need to check if BOTH j and k are selected
    
    # Better approach: for each node i, store a dict: for each other node j, 
    # store the set of k's such that (i,j,k) is bad
    # Then can_add(i) checks: for each j in selected, if k in selected for any k in conflicts[i][j]
    
    # Even better: use bitsets. For each node i, and each possible "partner" j,
    # store a bitset of k's. But this is complex.
    
    # Let's use a simpler but faster approach:
    # For each node i, store a list of (j, k) pairs where (i,j,k) is a bad triple
    # can_add(i): for each (j,k) in pairs[i], if j in selected AND k in selected, return False
    
    pairs_by_node = [[] for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].append((b, c))
        pairs_by_node[b].append((a, c))
        pairs_by_node[c].append((a, b))
    
    # Also build a quick lookup: for a pair (i,j), what are the k's that complete a bad triple?
    # triple_lookup[(i,j)] = set of k's
    triple_lookup = {}
    for a, b, c in bad_triples:
        for pair in [(a,b), (a,c), (b,c)]:
            if pair not in triple_lookup:
                triple_lookup[pair] = set()
            triple_lookup[pair].add(c if pair != (a,b) else (b if pair == (a,c) else a))
    
    # Simpler: just use the pairs_by_node approach with set-based selected
    # But optimize by precomputing for each node, a list of its conflict pairs sorted by frequency
    
    # For faster can_add, let's also build: for each node i, a bitset representation
    # Actually, let's just use the list approach but make it faster
    
    def can_add(selected, node):
        """Check if adding node to selected doesn't complete any bad triple."""
        for other1, other2 in pairs_by_node[node]:
            if other1 in selected and other2 in selected:
                return False
        return True
    
    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected = set()
        for node in order:
            if time.time() > deadline - 15:
                break
            if can_add(selected, node):
                selected.add(node)
        return selected
    
    def local_search(selected, max_iter=10000):
        """Local search: try to add unselected nodes, then try swaps."""
        selected = set(selected)
        unselected = set(range(n)) - selected
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 12:
                break
            
            # Phase 1: Try to add any unselected node
            for node in list(unselected):
                if time.time() > deadline - 12:
                    break
                if can_add(selected, node):
                    selected.add(node)
                    unselected.remove(node)
                    improved = True
            
            # Phase 2: Try single swaps - remove one selected, add one unselected
            if not improved:
                swap_found = False
                for s in list(selected):
                    if time.time() > deadline - 12:
                        break
                    if swap_found:
                        break
                    # Try removing s and adding each unselected node
                    selected.remove(s)
                    for u in list(unselected):
                        if time.time() > deadline - 12:
                            break
                        if can_add(selected, u):
                            selected.add(u)
                            unselected.remove(u)
                            swap_found = True
                            improved = True
                            break
                    # Re-add s if we didn't find a swap
                    if not swap_found:
                        selected.add(s)
                        unselected.discard(s)
            
            # Phase 3: Try double swaps - remove two selected, add two unselected
            if not improved and len(unselected) >= 2:
                double_swap_found = False
                selected_list = list(selected)
                # Limit the number of pairs to try
                max_pairs = min(len(selected_list) * (len(selected_list) - 1) // 2, 200)
                pair_count = 0
                for i in range(len(selected_list)):
                    if time.time() > deadline - 12:
                        break
                    if double_swap_found:
                        break
                    s1 = selected_list[i]
                    for j in range(i + 1, len(selected_list)):
                        if time.time() > deadline - 12:
                            break
                        if double_swap_found or pair_count >= max_pairs:
                            break
                        pair_count += 1
                        s2 = selected_list[j]
                        # Try removing s1 and s2
                        selected.remove(s1)
                        selected.remove(s2)
                        unselected.add(s1)
                        unselected.add(s2)
                        
                        # Try adding pairs from unselected
                        unselected_list = list(unselected)
                        for u1 in unselected_list:
                            if time.time() > deadline - 12:
                                break
                            if double_swap_found:
                                break
                            if can_add(selected, u1):
                                selected.add(u1)
                                unselected.remove(u1)
                                for u2 in unselected_list:
                                    if time.time() > deadline - 12:
                                        break
                                    if double_swap_found:
                                        break
                                    if u2 == u1:
                                        continue
                                    if can_add(selected, u2):
                                        selected.add(u2)
                                        unselected.remove(u2)
                                        double_swap_found = True
                                        improved = True
                                        break
                                # Undo if we only added one
                                if not double_swap_found:
                                    selected.discard(u1)
                                    unselected.add(u1)
                        
                        # Restore if no swap found
                        if not double_swap_found:
                            selected.add(s1)
                            selected.add(s2)
                            unselected.discard(s1)
                            unselected.discard(s2)
        
        return selected
    
    # Multiple restarts with different orderings
    best_solution = set()
    best_size = 0
    
    # Ordering 1: by conflict count (ascending) - less constrained nodes first
    conflict_count = [len(pairs_by_node[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda i: conflict_count[i])
    sol1 = greedy_init(order1)
    sol1 = local_search(sol1, max_iter=5000)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_solution = sol1
    
    # Ordering 2: by conflict count (descending) - more constrained nodes first
    order2 = sorted(range(n), key=lambda i: -conflict_count[i])
    sol2 = greedy_init(order2)
    sol2 = local_search(sol2, max_iter=5000)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_solution = sol2
    
    # Ordering 3: natural order
    order3 = list(range(n))
    sol3 = greedy_init(order3)
    sol3 = local_search(sol3, max_iter=5000)
    if len(sol3) > best_size:
        best_size = len(sol3)
        best_solution = sol3
    
    # Ordering 4: reverse natural
    order4 = list(range(n - 1, -1, -1))
    sol4 = greedy_init(order4)
    sol4 = local_search(sol4, max_iter=5000)
    if len(sol4) > best_size:
        best_size = len(sol4)
        best_solution = sol4
    
    # Ordering 5: odds first, then evens
    order5 = list(range(0, n, 2)) + list(range(1, n, 2))
    sol5 = greedy_init(order5)
    sol5 = local_search(sol5, max_iter=5000)
    if len(sol5) > best_size:
        best_size = len(sol5)
        best_solution = sol5
    
    # Additional local search passes on best solution
    if time.time() < deadline - 25:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            improved_sol = local_search(best_solution, max_iter=3000)
            if len(improved_sol) > len(best_solution):
                best_solution = improved_sol
    
    # Final attempt: try to add any remaining nodes
    if time.time() < deadline - 5:
        for node in range(n):
            if node not in best_solution and can_add(best_solution, node):
                best_solution.add(node)
    
    return [i + 1 for i in sorted(best_solution)]

# EVOLVE-BLOCK-END
