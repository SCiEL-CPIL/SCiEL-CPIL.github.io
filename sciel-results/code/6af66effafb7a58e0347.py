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
    deadline = start_time + 88  # 88s safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    rng = random.Random(42)
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if time.time() > deadline - 25:
            break
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    if not bad_triples:
        return elements
    
    # Build conflict structures using bitmasks for speed
    conflict_mask = [0] * n
    pairs_by_node = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add((b, c))
        pairs_by_node[b].add((a, c))
        pairs_by_node[c].add((a, b))
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
    
    conflict_count = [len(pairs_by_node[i]) for i in range(n)]
    
    # Precompute pair masks for each node
    pair_masks = [[] for _ in range(n)]
    for node in range(n):
        for other1, other2 in pairs_by_node[node]:
            pair_masks[node].append((1 << other1) | (1 << other2))
    
    def can_add(selected_mask, node):
        """Check if adding node to selected doesn't complete any bad triple."""
        for mask in pair_masks[node]:
            if (selected_mask & mask) == mask:
                return False
        return True
    
    def count_conflicts(mask):
        """Count number of bad triples fully contained in mask."""
        count = 0
        for a, b, c in bad_triples:
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                count += 1
        return count
    
    def greedy_init(order):
        """Greedy initialization with given order of nodes."""
        selected_mask = 0
        for node in order:
            if time.time() > deadline - 12:
                break
            if can_add(selected_mask, node):
                selected_mask |= (1 << node)
        return selected_mask
    
    def local_search(selected_mask, max_iter=5000):
        """Local search: try to add unselected nodes, then try 1-for-1 and 1-for-2 swaps."""
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 18:
                break
            
            # Phase 1: Try to add any unselected node
            for node in range(n):
                if time.time() > deadline - 18:
                    break
                if not (selected_mask & (1 << node)):
                    if can_add(selected_mask, node):
                        selected_mask |= (1 << node)
                        improved = True
            
            # Phase 2: Try 1-for-1 swaps (remove one, add one different)
            if not improved:
                swap_found = False
                for s in range(n):
                    if time.time() > deadline - 18:
                        break
                    if swap_found:
                        break
                    if not (selected_mask & (1 << s)):
                        continue
                    new_mask = selected_mask & ~(1 << s)
                    for u in range(n):
                        if time.time() > deadline - 18:
                            break
                        if swap_found:
                            break
                        if (new_mask & (1 << u)):
                            continue
                        if can_add(new_mask, u):
                            selected_mask = new_mask | (1 << u)
                            swap_found = True
                            improved = True
                            break
            
            # Phase 3: Try 1-for-2 swaps (remove one, add two)
            if not improved:
                swap_found = False
                for s in range(n):
                    if time.time() > deadline - 18:
                        break
                    if swap_found:
                        break
                    if not (selected_mask & (1 << s)):
                        continue
                    new_mask = selected_mask & ~(1 << s)
                    for u1 in range(n):
                        if time.time() > deadline - 18:
                            break
                        if swap_found:
                            break
                        if (new_mask & (1 << u1)):
                            continue
                        if not can_add(new_mask, u1):
                            continue
                        mask_with_u1 = new_mask | (1 << u1)
                        for u2 in range(u1 + 1, n):
                            if time.time() > deadline - 18:
                                break
                            if (mask_with_u1 & (1 << u2)):
                                continue
                            if can_add(mask_with_u1, u2):
                                selected_mask = mask_with_u1 | (1 << u2)
                                swap_found = True
                                improved = True
                                break
            
            # Phase 4: Try 2-for-1 swaps (remove two, add one) - can sometimes help
            if not improved and iter_count % 10 == 0:
                for s1 in range(n):
                    if time.time() > deadline - 18:
                        break
                    if not (selected_mask & (1 << s1)):
                        continue
                    for s2 in range(s1 + 1, n):
                        if time.time() > deadline - 18:
                            break
                        if not (selected_mask & (1 << s2)):
                            continue
                        new_mask = selected_mask & ~(1 << s1) & ~(1 << s2)
                        for u in range(n):
                            if time.time() > deadline - 18:
                                break
                            if (new_mask & (1 << u)):
                                continue
                            if can_add(new_mask, u):
                                selected_mask = new_mask | (1 << u)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return selected_mask
    
    def randomized_local_search(initial_mask, max_iter=3000):
        """Randomized local search with occasional random restarts of the selection."""
        current_mask = initial_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        temp = 0.5
        for it in range(max_iter):
            if time.time() > deadline - 18:
                break
            
            # Pick a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try adding a random unselected node
                unselected = [i for i in range(n) if not (current_mask & (1 << i))]
                if not unselected:
                    continue
                node = rng.choice(unselected)
                if can_add(current_mask, node):
                    new_mask = current_mask | (1 << node)
                    new_size = bin(new_mask).count('1')
                    if new_size >= current_size:
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
            elif move_type < 0.6:
                # Try removing a random selected node and adding a random unselected
                selected = [i for i in range(n) if (current_mask & (1 << i))]
                unselected = [i for i in range(n) if not (current_mask & (1 << i))]
                if not selected or not unselected:
                    continue
                s = rng.choice(selected)
                u = rng.choice(unselected)
                new_mask = (current_mask & ~(1 << s)) | (1 << u)
                if can_add(current_mask & ~(1 << s), u):
                    new_size = bin(new_mask).count('1')
                    if new_size >= current_size or rng.random() < temp:
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
            else:
                # Try removing a random selected node
                selected = [i for i in range(n) if (current_mask & (1 << i))]
                if not selected:
                    continue
                s = rng.choice(selected)
                new_mask = current_mask & ~(1 << s)
                new_size = bin(new_mask).count('1')
                if rng.random() < temp:
                    current_mask = new_mask
                    current_size = new_size
            
            temp *= 0.995
        
        # Final local search polish
        best_mask = local_search(best_mask, max_iter=500)
        return best_mask, bin(best_mask).count('1')
    
    # Multiple restarts with different orderings
    best_mask = 0
    best_size = 0
    
    # Ordering 1: by conflict count (ascending)
    order1 = sorted(range(n), key=lambda i: conflict_count[i])
    sol1 = greedy_init(order1)
    sol1 = local_search(sol1, max_iter=3000)
    if bin(sol1).count('1') > best_size:
        best_size = bin(sol1).count('1')
        best_mask = sol1
    
    # Ordering 2: reverse conflict count
    order2 = list(reversed(order1))
    sol2 = greedy_init(order2)
    sol2 = local_search(sol2, max_iter=3000)
    if bin(sol2).count('1') > best_size:
        best_size = bin(sol2).count('1')
        best_mask = sol2
    
    # Ordering 3: natural order
    order3 = list(range(n))
    sol3 = greedy_init(order3)
    sol3 = local_search(sol3, max_iter=3000)
    if bin(sol3).count('1') > best_size:
        best_size = bin(sol3).count('1')
        best_mask = sol3
    
    # Ordering 4: reverse natural
    order4 = list(range(n - 1, -1, -1))
    sol4 = greedy_init(order4)
    sol4 = local_search(sol4, max_iter=3000)
    if bin(sol4).count('1') > best_size:
        best_size = bin(sol4).count('1')
        best_mask = sol4
    
    # Ordering 5: random
    order5 = list(range(n))
    rng.shuffle(order5)
    sol5 = greedy_init(order5)
    sol5 = local_search(sol5, max_iter=3000)
    if bin(sol5).count('1') > best_size:
        best_size = bin(sol5).count('1')
        best_mask = sol5
    
    # Randomized local search from best solution
    if time.time() < deadline - 25:
        for _ in range(8):
            if time.time() > deadline - 20:
                break
            # Perturb best solution
            perturbed = best_mask
            # Remove 3-5 random elements
            selected = [i for i in range(n) if (perturbed & (1 << i))]
            if selected:
                k = min(rng.randint(3, 5), len(selected))
                to_remove = rng.sample(selected, k)
                for s in to_remove:
                    perturbed &= ~(1 << s)
            # Re-add greedily
            perturbed = local_search(perturbed, max_iter=2000)
            if bin(perturbed).count('1') > best_size:
                best_size = bin(perturbed).count('1')
                best_mask = perturbed
    
    # Additional local search passes on best solution
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            improved_mask = local_search(best_mask, max_iter=2000)
            if bin(improved_mask).count('1') > best_size:
                best_size = bin(improved_mask).count('1')
                best_mask = improved_mask
    
    # Final attempt: try to add any remaining nodes
    if time.time() < deadline - 5:
        for node in range(n):
            if not (best_mask & (1 << node)):
                if can_add(best_mask, node):
                    best_mask |= (1 << node)
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]

# EVOLVE-BLOCK-END
