import random
import time
import numpy as np

def construct():
    n = 50
    num_colors = 4
    start_time = time.time()
    time_limit = 115.0

    # Precompute edge indices and triangle neighbors
    # edge_to_ij: list of (i, j) for each edge index
    # tri_neighbors: for each edge index, list of (k, idx_ik, idx_jk)
    # We will use a 2D numpy array for the color matrix for fast access.
    # M[i, j] = color of edge (i, j)
    
    edge_to_ij = []
    idx_map = {}
    for i in range(n):
        for j in range(i+1, n):
            idx = len(edge_to_ij)
            edge_to_ij.append((i, j))
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
    
    num_edges = len(edge_to_ij)
    
    # Precompute for each edge e, the list of (k, idx_ik, idx_jk)
    # This allows fast calculation of delta when changing edge e's color.
    tri_neighbors = [[] for _ in range(num_edges)]
    for e, (i, j) in enumerate(edge_to_ij):
        for k in range(n):
            if k == i or k == j:
                continue
            if i < k:
                idx_ik = idx_map[(i, k)]
            else:
                idx_ik = idx_map[(k, i)]
            if j < k:
                idx_jk = idx_map[(j, k)]
            else:
                idx_jk = idx_map[(k, j)]
            tri_neighbors[e].append((k, idx_ik, idx_jk))

    # Fallback 32-vertex construction
    def get_fallback_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors_32 = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors_32}

    # Local Search State
    # M is n x n numpy array of colors. M[i,j] = M[j,i]. M[i,i] = -1 (ignored).
    # bad_edges_count[e] = number of mono-triangles involving edge e.
    # total_mono = total number of mono-triangles.
    
    for attempt in range(1000): # Increased restarts
        if time.time() - start_time > time_limit:
            break
        
        # Initialize random coloring
        # Use numpy for fast init
        M = np.random.randint(0, 4, (n, n), dtype=np.int8)
        np.fill_diagonal(M, -1)
        # Make symmetric
        M = np.triu(M, 1) + np.triu(M, 1).T
        
        # Initialize counts
        bad_edges_count = np.zeros(num_edges, dtype=np.int32)
        total_mono = 0
        
        # Calculate initial bad_edges_count and total_mono
        # For each edge e, check all k in tri_neighbors[e]
        for e in range(num_edges):
            i, j = edge_to_ij[e]
            c_e = M[i, j]
            count_e = 0
            for k, idx_ik, idx_jk in tri_neighbors[e]:
                # We need to get colors of (i,k) and (j,k)
                # Note: idx_ik and idx_jk are edge indices.
                # We can get the colors from M using the vertices stored in tri_neighbors?
                # No, tri_neighbors stores (k, idx_ik, idx_jk).
                # It's faster to just use M[i,k] and M[j,k] directly if we have i,j,k.
                # Let's re-evaluate.
                # tri_neighbors[e] has (k, idx_ik, idx_jk).
                # We can just use M[i, k] and M[j, k].
                # Wait, M is the matrix. So M[i,k] is O(1).
                # So we don't need idx_ik, idx_jk for counting if we have i, j, k.
                # But we need them for updating if we change an edge?
                # If we change edge e (i,j), we update bad_edges_count for e and for all edges (i,k) and (j,k).
                # To update bad_edges_count[idx_ik], we need to know the new count for that edge.
                # The count for an edge (u,v) is the number of w such that M[u,v]==M[u,w]==M[v,w].
                # So yes, we need to iterate over neighbors.
                
                if M[i, k] == c_e and M[j, k] == c_e:
                    count_e += 1
            bad_edges_count[e] = count_e
            total_mono += count_e
        
        total_mono //= 3
        
        # List of bad edges (edges with count > 0)
        # We maintain a list of indices of bad edges for random selection.
        # Since updating counts changes which edges are bad, we need to update this list.
        # A simple approach: keep a set or list, and when an edge's count goes 0->1, add it. 1->0, remove it.
        # Removal from a list is O(N). For 1225 edges, this is acceptable.
        # Or, just pick a random edge from all edges, and if it's not bad, skip?
        # No, that's inefficient if few edges are bad.
        # Let's maintain a list `bad_list`.
        
        bad_list = [e for e in range(num_edges) if bad_edges_count[e] > 0]
        
        # Local search loop
        # We iterate until time limit or no bad edges.
        while bad_list and (time.time() - start_time < time_limit):
            
            # Pick a random bad edge
            e = random.choice(bad_list)
            i, j = edge_to_ij[e]
            old_color = M[i, j]
            
            # Try all other colors
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                # Calculate delta for changing edge e from old_color to new_color
                delta = 0
                for k, idx_ik, idx_jk in tri_neighbors[e]:
                    c_ik = M[i, k]
                    c_jk = M[j, k]
                    
                    was_mono = (c_ik == old_color and c_jk == old_color)
                    is_mono = (c_ik == new_color and c_jk == new_color)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            # If we found an improvement or equal, accept it.
            # If best_delta > 0, we are in a local minimum with respect to single edge flips.
            # We should restart or do a random jump.
            
            if best_delta <= 0 and best_new_color != -1:
                # Apply change
                M[i, j] = best_new_color
                M[j, i] = best_new_color
                
                # Update counts
                # 1. Update count for edge e
                old_count_e = bad_edges_count[e]
                # Recalculate count for e? Or use delta?
                # Delta is the change in total_mono * 3?
                # No, delta calculated above is the change in the sum of counts over all edges?
                # Let's trace:
                # For each k, we checked if (i,j,k) was mono.
                # If it was mono before and is not now, delta -= 1.
                # This delta is the change in the number of mono-triangles incident to edge e?
                # No. A mono-triangle (i,j,k) is counted in bad_edges_count[e], bad_edges_count[idx_ik], and bad_edges_count[idx_jk].
                # So if (i,j,k) stops being mono, bad_edges_count[e] decreases by 1, bad_edges_count[idx_ik] decreases by 1, bad_edges_count[idx_jk] decreases by 1.
                # Total mono triangles decreases by 1.
                # The `delta` variable calculated above sums the changes for edge e's perspective.
                # Wait, the loop `for k in tri_neighbors[e]` checks triangles (i,j,k).
                # `was_mono` is true if (i,j,k) was mono.
                # If `was_mono` and not `is_mono`, the triangle is broken.
                # This contributes -1 to the total number of mono-triangles.
                # So `delta` is indeed the change in total_mono.
                
                # However, we need to update `bad_edges_count` for e, and for all neighbors idx_ik, idx_jk.
                
                # Update bad_edges_count[e]
                # New count for e:
                new_count_e = 0
                for k, idx_ik, idx_jk in tri_neighbors[e]:
                    if M[i, k] == best_new_color and M[j, k] == best_new_color:
                        new_count_e += 1
                bad_edges_count[e] = new_count_e
                
                # Update bad_list for e
                if old_count_e > 0 and new_count_e == 0:
                    # Remove e from bad_list
                    try:
                        bad_list.remove(e)
                    except ValueError:
                        pass
                elif old_count_e == 0 and new_count_e > 0:
                    # Add e to bad_list
                    bad_list.append(e)
                
                # Update neighbors
                # For each k, the edge (i,k) and (j,k) are affected.
                # We need to update bad_edges_count[idx_ik] and bad_edges_count[idx_jk].
                # To do this efficiently, we can recalculate the count for those edges.
                # Since n=50, recalculation is O(n).
                # There are n-2 neighbors, so O(n^2) per step. 2500 ops.
                # This is acceptable.
                
                for k, idx_ik, idx_jk in tri_neighbors[e]:
                    # Update idx_ik
                    u, v = edge_to_ij[idx_ik]
                    old_count_ik = bad_edges_count[idx_ik]
                    new_count_ik = 0
                    c_ik = M[u, v]
                    # Iterate over neighbors of (u,v)
                    # We need a list of neighbors for idx_ik.
                    # We have tri_neighbors[idx_ik].
                    for w, idx_uw, idx_vw in tri_neighbors[idx_ik]:
                        if M[u, w] == c_ik and M[v, w] == c_ik:
                            new_count_ik += 1
                    bad_edges_count[idx_ik] = new_count_ik
                    if old_count_ik > 0 and new_count_ik == 0:
                        try: bad_list.remove(idx_ik)
                        except ValueError: pass
                    elif old_count_ik == 0 and new_count_ik > 0:
                        bad_list.append(idx_ik)
                    
                    # Update idx_jk
                    u, v = edge_to_ij[idx_jk]
                    old_count_jk = bad_edges_count[idx_jk]
                    new_count_jk = 0
                    c_jk = M[u, v]
                    for w, idx_uw, idx_vw in tri_neighbors[idx_jk]:
                        if M[u, w] == c_jk and M[v, w] == c_jk:
                            new_count_jk += 1
                    bad_edges_count[idx_jk] = new_count_jk
                    if old_count_jk > 0 and new_count_jk == 0:
                        try: bad_list.remove(idx_jk)
                        except ValueError: pass
                    elif old_count_jk == 0 and new_count_jk > 0:
                        bad_list.append(idx_jk)
                        
                # Update total_mono
                total_mono += best_delta
                
            else:
                # Stuck: no improving move found.
                # Random jump: recolor 5 random edges
                for _ in range(5):
                    re = random.randint(0, num_edges - 1)
                    ri, rj = edge_to_ij[re]
                    rc = random.randint(0, 3)
                    
                    # We need to update counts for this change too.
                    # This is complex. Let's just reset the counts for the whole board?
                    # O(N^3) reset is ~125k ops. 100 jumps = 12.5M ops.
                    # This is likely too slow if we jump often.
                    
                    # Alternative: Just restart the whole search if stuck.
                    # Since we have many restarts, this is a valid strategy.
                    break
                
                break # Restart

        if not bad_list:
            # Found valid coloring
            # Extract colors in lexicographic order
            colors = []
            for e in range(num_edges):
                i, j = edge_to_ij[e]
                colors.append(int(M[i, j]))
            return {'n': n, 'colors': colors}
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    # Fallback
    return get_fallback_32()
