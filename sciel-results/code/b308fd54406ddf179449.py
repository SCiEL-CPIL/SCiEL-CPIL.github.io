"""Optimized program for finding max subset of [1,128] avoiding equal-lcm triples."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
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
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements

    # Build conflict pairs for each node
    # conflicts[i] = list of (j, k) such that (i, j, k) is a bad triple
    conflicts = [[] for _ in range(n)]
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Sort conflicts by frequency or just keep as is? 
    # Sorting by the "degree" of the other nodes might help pruning, but let's keep it simple.
    
    def can_add(selected, node):
        """Check if adding node to selected doesn't complete any bad triple."""
        for other1, other2 in conflicts[node]:
            if other1 in selected and other2 in selected:
                return False
        return True

    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected = set()
        for node in order:
            if can_add(selected, node):
                selected.add(node)
        return selected

    def local_search(selected, max_iter=10000, time_limit=None):
        """Local search: try to add unselected nodes, then try swaps."""
        selected = set(selected)
        unselected = set(range(n)) - selected
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time_limit and time.time() > time_limit:
                break
            
            # Phase 1: Try to add any unselected node
            # We iterate through unselected nodes. To avoid bias, we can shuffle or just iterate.
            # Iterating in a fixed order might miss opportunities if the set is dense.
            # But checking all is O(N * E).
            for node in list(unselected):
                if time_limit and time.time() > time_limit:
                    break
                if can_add(selected, node):
                    selected.add(node)
                    unselected.remove(node)
                    improved = True
            
            # Phase 2: Try single swaps - remove one selected, add one unselected
            # This is rarely beneficial for maximizing size (1 for 1), but might enable further additions?
            # Actually, removing 1 and adding 1 keeps size same. 
            # The goal is to increase size. So we look for:
            # 1. Add 1 (size +1)
            # 2. Remove 1, Add 2 (size +1)
            # 3. Remove 2, Add 3 (size +1)
            
            if not improved:
                # Try Remove 1, Add 2
                swap_found = False
                # We need to be careful with the order of operations to not break the set.
                # Iterate over selected nodes to remove
                for s in list(selected):
                    if time_limit and time.time() > time_limit:
                        break
                    if swap_found:
                        break
                    
                    selected.remove(s)
                    unselected.add(s)
                    
                    # Try to add two nodes from unselected
                    # We can't just add any two, they must be compatible with each other and the rest.
                    # Let's try to find a pair (u1, u2) in unselected such that:
                    # 1. u1 can be added to (selected \ {s})
                    # 2. u2 can be added to (selected \ {s} + {u1})
                    
                    # Optimization: Find candidates for u1
                    candidates_u1 = []
                    for u1 in unselected:
                        if can_add(selected, u1):
                            candidates_u1.append(u1)
                    
                    for u1 in candidates_u1:
                        if time_limit and time.time() > time_limit:
                            break
                        if swap_found:
                            break
                        
                        # Tentatively add u1
                        selected.add(u1)
                        
                        # Now try to add u2
                        for u2 in unselected:
                            if u2 == u1:
                                continue
                            if can_add(selected, u2):
                                # Success!
                                selected.add(u2)
                                unselected.remove(u1)
                                unselected.remove(u2)
                                swap_found = True
                                improved = True
                                break
                        
                        # Undo u1 if we didn't find u2
                        if not swap_found:
                            selected.remove(u1)
                            unselected.add(u1)
                    
                    # If we didn't find a swap, restore s
                    if not swap_found:
                        selected.add(s)
                        unselected.remove(s)
            
            # Phase 3: Kick / Random Restart within Local Search
            # If stuck, randomly remove a few nodes and try to re-add
            if not improved and len(unselected) > 0:
                if time_limit and time.time() > time_limit:
                    break
                # Remove 3 random nodes
                remove_count = min(3, len(selected))
                if remove_count > 0:
                    to_remove = random.sample(list(selected), remove_count)
                    for r in to_remove:
                        selected.remove(r)
                        unselected.add(r)
                    # The next iteration of Phase 1 will try to add nodes
                    # This effectively resets the local search state slightly
                    improved = True # Force another iteration

        return selected

    best_solution = set()
    best_size = 0
    
    # Multiple restarts with different orderings
    # 1. Natural order
    sol = greedy_init(list(range(n)))
    sol = local_search(sol, time_limit=deadline - 10)
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = sol

    # 2. Reverse natural
    sol = greedy_init(list(range(n-1, -1, -1)))
    sol = local_search(sol, time_limit=deadline - 10)
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = sol

    # 3. Randomized restarts
    # Use remaining time for random restarts
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        # Random order
        order = list(range(n))
        random.shuffle(order)
        sol = greedy_init(order)
        
        # Allocate time based on remaining time
        remaining = deadline - 5 - time.time()
        # If we have lots of time, do more local search
        # If little, do less
        time_limit = deadline - 2
        sol = local_search(sol, time_limit=time_limit)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            # If we hit 128, we can stop
            if best_size == n:
                break

    # Final polish: try to add any remaining nodes
    if time.time() < deadline - 1:
        for node in range(n):
            if node not in best_solution and can_add(best_solution, node):
                best_solution.add(node)

    return [i + 1 for i in sorted(best_solution)]

# EVOLVE-BLOCK-END
