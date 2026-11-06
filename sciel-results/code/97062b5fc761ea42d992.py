import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 12
    total_colorings = 1 << n
    all_ones = (1 << total_colorings) - 1

    # Precompute M0 and M1
    M0 = [0] * n
    M1 = [0] * n

    for c in range(total_colorings):
        for v in range(n):
            if (c >> v) & 1:
                M1[v] |= (1 << c)
            else:
                M0[v] |= (1 << c)

    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)

    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
        m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
        edge_masks.append(m0 | m1)

    best_count = float('inf')
    best_edges = None

    all_indices = list(range(num_edges))
    
    # Run multiple restarts
    # n=12 is small, we can afford many restarts
    restarts = 200 

    for r in range(restarts):
        if time.time() > deadline - 5:
            break

        random.seed(42 + r)

        # 1. Greedy Set Cover to get initial solution
        covered = 0
        selected_indices = []
        remaining = all_indices[:]
        random.shuffle(remaining)

        while covered != all_ones and remaining:
            best_gain = -1
            best_idx = -1
            for idx in remaining:
                gain = (edge_masks[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain == 0:
                break
            selected_indices.append(best_idx)
            covered |= edge_masks[best_idx]
            remaining.remove(best_idx)

        if covered != all_ones:
            continue

        # 2. Local Search with Directed Swaps
        # Maintain count of coverings for each coloring
        count = [0] * total_colorings
        for idx in selected_indices:
            m = edge_masks[idx]
            # Iterate set bits in m
            temp = m
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                count[c] += 1
                temp ^= lsb

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Deletion Phase
            # Shuffle indices to randomize order
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            deleted_any = False
            for i in indices_to_check:
                if i >= len(selected_indices):
                    continue
                e_idx = selected_indices[i]
                m = edge_masks[e_idx]
                
                # Check if any coloring in m has count 1
                # If so, removing e_idx would make it 0 (invalid)
                can_remove = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    if count[c] == 1:
                        can_remove = False
                        break
                    temp ^= lsb
                
                if can_remove:
                    # Remove e_idx
                    selected_indices.pop(i)
                    # Update counts
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        count[c] -= 1
                        temp ^= lsb
                    improved = True
                    deleted_any = True
                    break # Restart deletion loop after a removal

            if deleted_any:
                continue # Go back to top of while loop to try more deletions

            # Swap Phase
            # If no deletion possible, try a directed swap
            # Pick a random edge to remove
            if selected_indices:
                e_pos = random.randint(0, len(selected_indices) - 1)
                e_idx = selected_indices[e_pos]
                m_e = edge_masks[e_idx]
                
                # Deficit: colorings that would become 0 if e_idx is removed
                # These are colorings c in m_e where count[c] == 1
                deficit = 0
                temp = m_e
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    if count[c] == 1:
                        deficit |= lsb
                    temp ^= lsb
                
                if deficit == 0:
                    # Should have been removed in deletion phase
                    continue

                # Find an edge f not in selected_indices such that mask(f) covers deficit
                # i.e., mask(f) & deficit == deficit
                # To speed this up, we can just iterate through all unselected edges
                # Since num_edges is 792, this is fast.
                
                # Create a set of selected indices for O(1) lookup? 
                # selected_indices is a list. For 792 edges, linear search is okay, but set is better.
                # However, we modify selected_indices.
                # Let's just iterate and check membership.
                
                found_swap = False
                # Shuffle unselected edges to randomize which swap we pick
                unselected = [i for i in range(num_edges) if i not in selected_indices]
                random.shuffle(unselected)
                
                for f_idx in unselected:
                    m_f = edge_masks[f_idx]
                    if (m_f & deficit) == deficit:
                        # Valid swap
                        # Remove e_idx, add f_idx
                        selected_indices[e_pos] = f_idx
                        
                        # Update counts
                        # Remove e_idx contributions
                        temp = m_e
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] -= 1
                            temp ^= lsb
                            
                        # Add f_idx contributions
                        temp = m_f
                        while temp:
                            lsb = temp & -temp
                            c = lsb.bit_length() - 1
                            count[c] += 1
                            temp ^= lsb
                            
                        improved = True
                        found_swap = True
                        break # Restart deletion phase after swap

                if not found_swap:
                    # No swap found for this e_idx. Try another e_idx?
                    # Or restart local search?
                    # To avoid getting stuck, we can just break out of this local search instance
                    # and rely on the next restart.
                    # Or we can try a random "kick" - but that's complex.
                    # Given the restarts, breaking is fine.
                    pass

        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_edges = [list(all_edges[i]) for i in selected_indices]
            
            if best_count <= 51:
                # If we hit the known minimum, we can stop early 
                # but we might want to verify if 50 is possible.
                # 51 is the known best. 50 is unlikely.
                # If we find 51, we can break to save time for other restarts?
                # No, other restarts might find 50.
                # But if we find 51, we are good.
                if best_count < 51:
                    break

    if best_edges is None:
        # Fallback
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    else:
        best_n = n

    return {'n': best_n, 'edges': best_edges}
