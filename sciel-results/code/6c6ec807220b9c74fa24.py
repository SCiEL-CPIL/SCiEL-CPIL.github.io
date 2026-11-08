"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate all weight-4 vectors with entries +/-1
    w4_list = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            w4_list.append(v)
    
    w4_arr = np.array(w4_list, dtype=np.int64)
    n = len(w4_arr)  # 5280
    
    # Precompute dot product matrix
    dots = w4_arr @ w4_arr.T
    
    # Conflict: dot > 2 means distance < 2*norm = 4, but we need distance >= norm
    # norm = sqrt(4) = 2, so distance >= 2 means dot <= 4 - 4/2 = 2
    # Actually: distance^2 = |a|^2 + |b|^2 - 2*a.b = 4 + 4 - 2*a.b = 8 - 2*a.b
    # We need distance >= 2, so distance^2 >= 4
    # 8 - 2*a.b >= 4 => a.b <= 2
    # So conflict is a.b > 2
    conflict = (dots > 2).astype(np.int8)
    
    # Maximum weight independent set via branch-and-bound
    # Use minimum degree heuristic for ordering
    
    # Compute conflict counts
    conflict_count = conflict.sum(axis=1)
    
    # Order by minimum degree (most constrained first)
    order = np.argsort(conflict_count, kind='stable')
    
    # Remap to new indices
    new_to_old = order
    old_to_new = np.zeros(n, dtype=np.int64)
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    
    # Reorder conflict matrix
    conflict_ordered = conflict[np.ix_(order, order)]
    
    # Bitset representation for fast operations
    # Use Python integers as bitsets
    n_bits = n
    word_size = 64
    n_words = (n_bits + word_size - 1) // word_size
    
    # Convert conflict matrix to bitsets (one per vertex)
    # conflict_bits[i] is a bitset of vertices that conflict with i
    conflict_bits = []
    for i in range(n):
        bits = 0
        row = conflict_ordered[i]
        for j in range(n):
            if row[j]:
                bits |= (1 << j)
        conflict_bits.append(bits)
    
    # Greedy initial solution
    selected = set()
    available = (1 << n) - 1
    
    for i in order:
        if time.time() > deadline - 10:
            break
        # Check if i conflicts with any selected
        if selected and (conflict_bits[i] & (1 << i)) == 0:
            # Check against all selected
            conflict_with_selected = False
            for s in selected:
                if (conflict_bits[i] >> s) & 1:
                    conflict_with_selected = True
                    break
            if not conflict_with_selected:
                selected.add(i)
    
    best_solution = set(selected)
    best_size = len(selected)
    
    # Local search: 1-swap moves
    unselected = [i for i in range(n) if i not in selected]
    
    for _round in range(5):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for u in unselected:
            if time.time() > deadline - 5:
                break
            
            # Find conflicts of u with selected
            conflicting = [s for s in selected if (conflict_bits[u] >> s) & 1]
            
            if not conflicting:
                selected.add(u)
                improved = True
                continue
            
            # Try removing each conflicting vertex
            for s in conflicting:
                if time.time() > deadline - 5:
                    break
                
                # Check if u conflicts with selected - {s}
                remaining_conflict = False
                for t in selected:
                    if t == s:
                        continue
                    if (conflict_bits[u] >> t) & 1:
                        remaining_conflict = True
                        break
                
                if not remaining_conflict:
                    selected.remove(s)
                    selected.add(u)
                    improved = True
                    break
            
            if improved:
                unselected = [i for i in range(n) if i not in selected]
                break
        
        if not improved:
            break
    
    if len(selected) > best_size:
        best_solution = set(selected)
        best_size = len(selected)
    
    # 2-swap moves
    for _round in range(2):
        if time.time() > deadline - 3:
            break
        
        improved = False
        unselected = [i for i in range(n) if i not in selected]
        
        for u1 in unselected:
            if time.time() > deadline - 3:
                break
            
            for u2 in unselected:
                if u1 >= u2:
                    continue
                if time.time() > deadline - 3:
                    break
                
                # Check if u1 and u2 conflict
                if (conflict_bits[u1] >> u2) & 1:
                    continue
                
                # Find conflicts of u1 and u2 with selected
                conflicts_u1 = [s for s in selected if (conflict_bits[u1] >> s) & 1]
                conflicts_u2 = [s for s in selected if (conflict_bits[u2] >> s) & 1]
                
                all_conflicts = set(conflicts_u1) | set(conflicts_u2)
                
                if len(all_conflicts) <= 2:
                    # Try removing all conflicts
                    new_selected = set(selected) - all_conflicts
                    new_selected.add(u1)
                    new_selected.add(u2)
                    
                    # Verify no conflicts within new_selected
                    valid = True
                    for a in new_selected:
                        for b in new_selected:
                            if a < b and (conflict_bits[a] >> b) & 1:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        selected = new_selected
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    if len(selected) > best_size:
        best_solution = set(selected)
        best_size = len(selected)
    
    # Return the best solution
    result_indices = sorted(best_solution)
    return w4_arr[result_indices]

# EVOLVE-BLOCK-END
