"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute forbidden triples for each edge
    # Edge idx with direction i, for each j != i, the 4-face has:
    # (x, y) dir i, (x^j, y^j) dir i, (x, x^j) dir j, (y, y^j) dir j
    # So if we have (x,y), the other 3 edges are (x^j, y^j), (x, x^j), (y, y^j)
    forbidden = [[] for _ in range(m)]
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e1 = (min(x, c), max(x, c))  # (x, x^j)
            e2 = (min(y, e), max(y, e))  # (y, y^j)
            e3 = (min(c, e), max(c, e))  # (x^j, y^j)
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                forbidden[idx].append((idx1, idx2, idx3))
    
    # Precompute for each edge, which other edges it "conflicts" with in a C4
    # Edge a conflicts with edge b if they are in the same 4-face and not opposite
    # Actually, for C4-freeness: a C4 is formed when 4 edges of a face are all selected
    # So edge idx is "blocked" if there exists a triple (i1,i2,i3) in forbidden[idx]
    # such that all of i1,i2,i3 are selected
    
    def can_add(idx, chosen):
        for (i1, i2, i3) in forbidden[idx]:
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    # Initialize with a good heuristic: select edges based on a random 2-coloring of vertices
    # In a 2-coloring, edges between same-colored vertices form a C4-free set? No.
    # Better: use a random subset, then greedy complete
    
    # Try multiple random initializations and keep the best
    best_count = 0
    best_chosen = np.zeros(m, dtype=bool)
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        # Random initialization: each edge selected with probability 0.5
        chosen = np.random.random(m) < 0.5
        count = chosen.sum()
        
        # Greedy: remove edges that create C4s
        for idx in range(m):
            if chosen[idx]:
                for (i1, i2, i3) in forbidden[idx]:
                    if chosen[i1] and chosen[i2] and chosen[i3]:
                        # Remove one of the four to break the C4
                        # Remove idx itself for simplicity
                        chosen[idx] = False
                        count -= 1
                        break
        
        # Greedy completion: add all edges that can be added
        for idx in range(m):
            if not chosen[idx]:
                if can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
        
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
    
    # Local search: simulated annealing
    chosen = best_chosen.copy()
    count = best_count
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99998
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and T > T_min:
        # Random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random unselected edge
            unselected = [i for i in range(m) if not chosen[i]]
            if not unselected:
                continue
            idx = random.choice(unselected)
            if can_add(idx, chosen):
                chosen[idx] = True
                count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try swap: find a conflicting edge to remove
                conflicts = []
                for (i1, i2, i3) in forbidden[idx]:
                    if chosen[i1] and chosen[i2] and chosen[i3]:
                        conflicts.extend([i1, i2, i3])
                
                if conflicts:
                    remove_idx = random.choice(conflicts)
                    if chosen[remove_idx]:
                        chosen[remove_idx] = False
                        count -= 1
                        if can_add(idx, chosen):
                            chosen[idx] = True
                            count += 1
                            if count > best_count:
                                best_count = count
                                best_chosen = chosen.copy()
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            # Revert
                            chosen[idx] = False
                            chosen[remove_idx] = True
                            count += 1
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        elif move_type < 0.7:
            # Try to remove a random selected edge, then greedily add
            selected = [i for i in range(m) if chosen[i]]
            if not selected:
                continue
            idx = random.choice(selected)
            chosen[idx] = False
            count -= 1
            
            # Greedily add all possible edges
            for other_idx in range(m):
                if not chosen[other_idx]:
                    if can_add(other_idx, chosen):
                        chosen[other_idx] = True
                        count += 1
            
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
                no_improve = 0
            else:
                # Revert if worse
                if count < best_count - 3:
                    chosen = best_chosen.copy()
                    count = best_count
                no_improve += 1
        
        else:
            # 2-swap: remove two selected edges, add two unselected
            selected = [i for i in range(m) if chosen[i]]
            unselected = [i for i in range(m) if not chosen[i]]
            if len(selected) < 2 or len(unselected) < 2:
                continue
            
            r1, r2 = random.sample(selected, 2)
            a1, a2 = random.sample(unselected, 2)
            
            # Try the swap
            chosen[r1] = False
            chosen[r2] = False
            count -= 2
            
            success = True
            if can_add(a1, chosen):
                chosen[a1] = True
                count += 1
            else:
                success = False
            
            if success:
                if can_add(a2, chosen):
                    chosen[a2] = True
                    count += 1
                else:
                    success = False
            
            if not success:
                # Revert
                chosen[a1] = False
                chosen[a2] = False
                chosen[r1] = True
                chosen[r2] = True
                count = best_count if count < best_count else count
                # Recount
                count = chosen.sum()
                no_improve += 1
            else:
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        
        T *= cooling
        
        if no_improve > max_no_improve:
            # Reset to best and restart with higher temperature
            chosen = best_chosen.copy()
            count = best_count
            T = max(T, 0.5)
            no_improve = 0
    
    # Final greedy polish: multiple passes
    for _ in range(5):
        improved = False
        for idx in range(m):
            if not chosen[idx]:
                if can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    improved = True
        if not improved:
            break
    
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END

