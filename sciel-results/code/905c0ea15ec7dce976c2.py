"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # 1. Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                
    m = len(all_edges)  # 448
    
    # 2. Precompute all C4s (faces)
    # A C4 is defined by a vertex x and two distinct directions i, j.
    # Vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x^j, x^i^j), (x, x^j), (x^i, x^i^j)
    c4s = []
    seen_c4s = set()
    
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v3, v4), max(v3, v4))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v4), max(v2, v4))
                
                idx1 = edge_to_idx.get(e1)
                idx2 = edge_to_idx.get(e2)
                idx3 = edge_to_idx.get(e3)
                idx4 = edge_to_idx.get(e4)
                
                if None not in (idx1, idx2, idx3, idx4):
                    c4 = tuple(sorted([idx1, idx2, idx3, idx4]))
                    if c4 not in seen_c4s:
                        seen_c4s.add(c4)
                        c4s.append(c4)
                        
    num_c4s = len(c4s)
    
    # 3. Build incidence structures
    # edge_to_c4s[edge_idx] = list of c4 indices
    edge_to_c4s = [[] for _ in range(m)]
    # c4_to_edges[c4_idx] = list of 4 edge indices
    c4_to_edges = [list(c4) for c4 in c4s]
    
    for c4_idx, c4 in enumerate(c4s):
        for e in c4:
            edge_to_c4s[e].append(c4_idx)
            
    # 4. Spectral Initialization
    # Build conflict graph: edges are connected if they share a C4
    row = []
    col = []
    data = []
    for c4 in c4s:
        # All pairs in this C4 are connected in the conflict graph
        for i in range(4):
            for j in range(i + 1, 4):
                e1 = c4[i]
                e2 = c4[j]
                row.append(e1)
                col.append(e2)
                data.append(1.0)
                row.append(e2)
                col.append(e1)
                data.append(1.0)
                
    A = csr_matrix((data, (row, col)), shape=(m, m))
    L = laplacian(A, normed=True)
    
    try:
        # Find Fiedler vector
        eigenvalues, eigenvectors = eigsh(L, k=3, sigma=0, which='LM')
        idx_sort = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx_sort[1]]
    except Exception:
        fiedler = np.random.randn(m)
        fiedler -= fiedler.mean()
        
    # Initial selection: edges with positive fiedler value
    selected = np.zeros(m, dtype=bool)
    for i in range(m):
        if fiedler[i] > 0:
            selected[i] = True
            
    # Enforce validity: remove edges from any C4 that has 4 selected
    # We do this by iterating C4s and removing one if count is 4
    c4_count = np.zeros(num_c4s, dtype=int)
    for e in range(m):
        if selected[e]:
            for c4_idx in edge_to_c4s[e]:
                c4_count[c4_idx] += 1
                
    # Fix invalid C4s
    for c4_idx in range(num_c4s):
        if c4_count[c4_idx] == 4:
            # Remove the edge with the highest fiedler value (or random) to break the cycle
            # Actually, removing any one makes it valid.
            # Let's remove the one with the smallest fiedler value to keep the "stronger" ones?
            # Or just the first one.
            e_to_remove = c4_to_edges[c4_idx][0]
            selected[e_to_remove] = False
            c4_count[c4_idx] -= 1
            # Update c4_count for other C4s containing e_to_remove
            for other_c4_idx in edge_to_c4s[e_to_remove]:
                if other_c4_idx != c4_idx:
                    c4_count[other_c4_idx] -= 1
                    # Check if this makes other_c4_idx invalid? No, we only decreased count.
                    # But we might have created a new invalid state? No, 3 is valid.
                    
    # Recalculate blocked counts
    # blocked[e] = number of C4s containing e where c4_count == 3 and selected[e] is False
    blocked = np.zeros(m, dtype=int)
    for e in range(m):
        if not selected[e]:
            for c4_idx in edge_to_c4s[e]:
                if c4_count[c4_idx] == 3:
                    blocked[e] += 1
                    
    current_size = np.sum(selected)
    best_size = current_size
    best_selected = selected.copy()
    
    # 5. Local Search (Simulated Annealing)
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    
    # Helper to update state
    def update_state(e, is_add, c4_count, blocked):
        """
        Updates c4_count and blocked arrays after adding or removing edge e.
        Assumes selected[e] is already updated.
        """
        if is_add:
            # e is now selected
            for c4_idx in edge_to_c4s[e]:
                old_count = c4_count[c4_idx]
                c4_count[c4_idx] = old_count + 1
                new_count = c4_count[c4_idx]
                
                # Update blocked for edges in this C4
                for other_e in c4_to_edges[c4_idx]:
                    if other_e == e:
                        continue
                    if not selected[other_e]:
                        if old_count == 2 and new_count == 3:
                            blocked[other_e] += 1
                        # If old_count was 3, it was already blocked.
                        # If new_count is 4, it's invalid, but we prevent that.
        else:
            # e is now unselected
            for c4_idx in edge_to_c4s[e]:
                old_count = c4_count[c4_idx]
                c4_count[c4_idx] = old_count - 1
                new_count = c4_count[c4_idx]
                
                for other_e in c4_to_edges[c4_idx]:
                    if other_e == e:
                        continue
                    if not selected[other_e]:
                        if old_count == 3 and new_count == 2:
                            blocked[other_e] -= 1
                        # If old_count was 4, it was invalid. Now 3, so other_e is blocked?
                        # Wait, if old_count was 4, it means all 4 were selected.
                        # But we maintain validity, so old_count should never be 4.
                        # So we only transition 3->2 or 2->3.
                        
    # Precompute random list for shuffling if needed, or just use random.randint
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with higher temperature if stuck
            T = 5.0
            # Optional: perturb best solution
            # For now, just continue with cooling reset
            
        e = random.randint(0, m - 1)
        
        if selected[e]:
            # Try removing
            # Delta = -1
            # Accept with prob exp(1/T)
            if random.random() < np.exp(1.0 / T):
                # Perform removal
                selected[e] = False
                current_size -= 1
                update_state(e, False, c4_count, blocked)
                
                # Try to add back any unblocked edges? No, that's the next step.
                # But we can do a greedy polish after every few steps?
                # No, let's keep it simple.
                
        else:
            # Try adding
            if blocked[e] == 0:
                # Can add directly
                selected[e] = True
                current_size += 1
                update_state(e, True, c4_count, blocked)
                
                if current_size > best_size:
                    best_size = current_size
                    best_selected = selected.copy()
            else:
                # Try a swap
                # Find a blocking C4
                # blocked[e] > 0 means there is at least one C4 with count 3
                # Pick a random C4 from edge_to_c4s[e] that is blocking
                blocking_c4s = [c4_idx for c4_idx in edge_to_c4s[e] if c4_count[c4_idx] == 3]
                
                if blocking_c4s:
                    c4_idx = random.choice(blocking_c4s)
                    # Pick a random selected edge v in this C4 to remove
                    # The C4 has 3 selected edges and e (unselected)
                    selected_in_c4 = [oe for oe in c4_to_edges[c4_idx] if selected[oe]]
                    
                    if len(selected_in_c4) == 3:
                        v = random.choice(selected_in_c4)
                        
                        # Simulate removing v and adding e
                        # 1. Remove v
                        selected[v] = False
                        current_size -= 1
                        update_state(v, False, c4_count, blocked)
                        
                        # 2. Check if e can be added
                        if blocked[e] == 0:
                            # Add e
                            selected[e] = True
                            current_size += 1
                            update_state(e, True, c4_count, blocked)
                            
                            # Swap accepted (size unchanged, but structure changed)
                            # We don't update best_size here as size is same
                        else:
                            # Revert
                            # Remove e? No, e wasn't added.
                            # Re-add v
                            selected[v] = True
                            current_size += 1
                            update_state(v, True, c4_count, blocked)
                            
        T *= cooling
        
        # Occasional greedy polish to capture easy gains
        if random.random() < 0.01:
            # Try to add any unblocked edge
            for i in range(m):
                if not selected[i] and blocked[i] == 0:
                    selected[i] = True
                    current_size += 1
                    update_state(i, True, c4_count, blocked)
            if current_size > best_size:
                best_size = current_size
                best_selected = selected.copy()
                
    # Final greedy polish
    for i in range(m):
        if not selected[i] and blocked[i] == 0:
            selected[i] = True
            current_size += 1
            update_state(i, True, c4_count, blocked)
            
    if current_size > best_size:
        best_size = current_size
        best_selected = selected.copy()
        
    # Convert to output
    out = []
    for idx in range(m):
        if best_selected[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
            
    return out

# EVOLVE-BLOCK-END
