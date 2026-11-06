"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
import time
import itertools
import random

def _validate(coloring, n=16):
    """Check that every K4 spans at least 5 distinct colours."""
    col = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = coloring[idx]
            idx += 1
    for a, b, c, d in itertools.combinations(range(n), 4):
        colours = set()
        colours.add(col[a][b])
        colours.add(col[a][c])
        colours.add(col[a][d])
        colours.add(col[b][c])
        colours.add(col[b][d])
        colours.add(col[c][d])
        if len(colours) < 5:
            return False
    return True

def _count_colours(coloring):
    """Count distinct colours used."""
    return len(set(coloring))

def _validate_partial(col_matrix, n=16):
    """Check all K4s that are fully assigned. col_matrix is n x n with -1 for unassigned."""
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        colours = set()
        complete = True
        for x, y in edges:
            c_val = col_matrix[x][y]
            if c_val == -1:
                complete = False
                break
            colours.add(c_val)
        if complete and len(colours) < 5:
            return False
    return True

def _get_k4_edges(a, b, c, d):
    """Return list of edge indices for K4 on vertices a,b,c,d."""
    edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
    return edges

def _edge_index(i, j):
    """Get the index in the flat array for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: i*(i-1)/2
    # Then offset by j - i - 1
    return i*(i-1)//2 + (j - i - 1)

def _try_linear(a, b, k, n=16):
    """Try (a*i + b*j) % k for i<j."""
    coloring = [(a*i + b*j) % k for i in range(n) for j in range(i+1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_quadratic(c1, c2, c3, k, n=16):
    """Try (c1*i*j + c2*i + c3*j) % k for i<j."""
    coloring = [(c1*i*j + c2*i + c3*j) % k for i in range(n) for j in range(i+1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_grid_2d(k, n=16):
    """Map vertex i to (i%4, i//4), colour = (r_i+r_j + 4*(s_i+s_j)) % k."""
    coloring = []
    for i in range(n):
        ri, si = i % 4, i // 4
        for j in range(i+1, n):
            rj, sj = j % 4, j // 4
            coloring.append((ri + rj + 4*(si + sj)) % k)
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_grid_2d_v2(k, n=16):
    """Map vertex i to (i%4, i//4), colour = (ri*rj + si*sj + ri + sj) % k."""
    coloring = []
    for i in range(n):
        ri, si = i % 4, i // 4
        for j in range(i+1, n):
            rj, sj = j % 4, j // 4
            coloring.append((ri*rj + si*sj + ri + sj) % k)
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_xor_shift(k, n=16):
    """colour = (i ^ j) % k."""
    coloring = [(i ^ j) % k for i in range(n) for j in range(i+1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_xor_sum(k, n=16):
    """colour = ((i ^ j) + (i + j)) % k."""
    coloring = [((i ^ j) + (i + j)) % k for i in range(n) for j in range(i+1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _local_search(initial_coloring, n=16, max_iters=5000, deadline=None):
    """Try to reduce number of colours via local search."""
    coloring = list(initial_coloring)
    best_count = _count_colours(coloring)
    
    # Build adjacency matrix
    col_mat = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col_mat[i][j] = col_mat[j][i] = coloring[idx]
            idx += 1
    
    # Precompute all K4s and their edge indices
    k4s = list(itertools.combinations(range(n), 4))
    k4_edge_indices = []
    for a,b,c,d in k4s:
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        eidx = [_edge_index(x,y) for x,y in edges]
        k4_edge_indices.append(eidx)
    
    current_count = best_count
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Try to recolor each edge to a colour already in use (or remove a colour)
        for eidx in range(len(coloring)):
            if deadline is not None and time.time() > deadline:
                break
            # Find which edge this is
            # Decode eidx to (i,j)
            # i*(i-1)/2 + (j-i-1) = eidx
            i = 0
            while i*(i-1)//2 <= eidx:
                i += 1
            i -= 1
            j = i + 1 + (eidx - i*(i-1)//2)
            
            old_c = coloring[eidx]
            
            # Try all colours currently in use
            used_colours = set(coloring)
            for new_c in used_colours:
                if new_c == old_c:
                    continue
                # Check if recoloring this edge keeps all K4s valid
                # Only K4s containing edge (i,j) need checking
                valid = True
                for eidx_list in k4_edge_indices:
                    if eidx in eidx_list:
                        colours = set()
                        for ei in eidx_list:
                            if ei == eidx:
                                colours.add(new_c)
                            else:
                                colours.add(coloring[ei])
                        if len(colours) < 5:
                            valid = False
                            break
                if valid:
                    coloring[eidx] = new_c
                    col_mat[i][j] = col_mat[j][i] = new_c
                    new_count = _count_colours(coloring)
                    if new_count < current_count:
                        current_count = new_count
                        best_count = new_count
                        improved = True
                    break  # take the first valid recoloring
    
    return coloring, best_count

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_coloring = None
    best_count = 17
    
    # Baseline
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if _validate(baseline):
        best_coloring = baseline
        best_count = 17
    
    # Try smaller k with various forms
    for k in range(13, 17):
        if time.time() > deadline - 30:
            break
        # Linear forms
        for a in range(1, 6):
            for b in range(1, 6):
                if time.time() > deadline - 25:
                    break
                coloring, count = _try_linear(a, b, k)
                if coloring is not None and count < best_count:
                    best_coloring = coloring
                    best_count = count
        
        # Quadratic forms
        if time.time() > deadline - 20:
            break
        for c1 in range(0, 3):
            for c2 in range(0, 3):
                for c3 in range(0, 3):
                    if time.time() > deadline - 15:
                        break
                    coloring, count = _try_quadratic(c1, c2, c3, k)
                    if coloring is not None and count < best_count:
                        best_coloring = coloring
                        best_count = count
        
        # Grid forms
        if time.time() > deadline - 10:
            break
        coloring, count = _try_grid_2d(k)
        if coloring is not None and count < best_count:
            best_coloring = coloring
            best_count = count
        
        coloring, count = _try_grid_2d_v2(k)
        if coloring is not None and count < best_count:
            best_coloring = coloring
            best_count = count
        
        # XOR forms
        if time.time() > deadline - 5:
            break
        coloring, count = _try_xor_shift(k)
        if coloring is not None and count < best_count:
            best_coloring = coloring
            best_count = count
        
        coloring, count = _try_xor_sum(k)
        if coloring is not None and count < best_count:
            best_coloring = coloring
            best_count = count
    
    # If we found a valid coloring with >13 colours, try local search to reduce
    if best_coloring is not None and best_count > 13 and time.time() < deadline - 5:
        coloring, count = _local_search(best_coloring, deadline=deadline - 2)
        if count < best_count:
            best_coloring = coloring
            best_count = count
    
    if best_coloring is None:
        best_coloring = baseline
    
    return best_coloring
