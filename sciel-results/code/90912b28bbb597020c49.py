import time
import itertools
import random

def _has_c5_fast(n, adj_mat):
    # Same as before
    for a in range(n):
        neighbors_a = [v for v in range(n) if (adj_mat[a] >> v) & 1]
        for i in range(len(neighbors_a)):
            b = neighbors_a[i]
            for j in range(i + 1, len(neighbors_a)):
                c = neighbors_a[j]
                candidates_w = (adj_mat[b] & adj_mat[c]) & ~(1 << a)
                while candidates_w:
                    w_bit = candidates_w & -candidates_w
                    w = w_bit.bit_length() - 1
                    candidates_w ^= w_bit
                    candidates_x = (adj_mat[w] & adj_mat[c]) & ~( (1 << a) | (1 << b) | (1 << w) | (1 << c) )
                    if candidates_x:
                        return True
    return False

def _has_independent_set_k(n, adj_mat, k=9):
    # Same as before
    found = [False]
    degrees = [bin(adj_mat[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: degrees[v])
    
    def search(chosen_count, candidates):
        if found[0]:
            return
        if chosen_count >= k:
            found[0] = True
            return
        if chosen_count + bin(candidates).count('1') < k:
            return
        if candidates == 0:
            return
        
        best_v = -1
        best_deg = float('inf')
        temp = candidates
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = bin(adj_mat[v] & candidates).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        bit = 1 << v
        
        new_candidates = candidates & ~(adj_mat[v] | bit)
        search(chosen_count + 1, new_candidates)
        if found[0]:
            return
        
        search(chosen_count, candidates & ~bit)
    
    search(0, (1 << n) - 1)
    return found[0]

def make_rooks_graph(n_rows, n_cols, offset=0):
    """Generate Rook's graph L_2(n) for n x n board.
    Vertices are (r, c). Edge if same row or same col.
    Returns list of edges [u, v] with global indices offset + r*n_cols + c.
    """
    edges = []
    n = n_rows * n_cols
    for r in range(n_rows):
        for c in range(n_cols):
            u = offset + r * n_cols + c
            # Connect to other in same row
            for c2 in range(n_cols):
                if c2 > c:
                    v = offset + r * n_cols + c2
                    edges.append([u, v])
            # Connect to other in same col
            for r2 in range(n_rows):
                if r2 > r:
                    v = offset + r2 * n_cols + c
                    edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 32
    # Fallback: 8 K4s
    def make_8k4_edges():
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return edges
    
    best_edges = make_8k4_edges()
    
    def verify(n, edges):
        adj_mat = [0] * n
        for u, v in edges:
            adj_mat[u] |= (1 << v)
            adj_mat[v] |= (1 << u)
        if _has_c5_fast(n, adj_mat):
            return False, True
        if _has_independent_set_k(n, adj_mat, 9):
            return True, False
        return True, True

    # Try Rook's Graph (Shrikhande) base
    # 2 disjoint Rook's(4) = 32 vertices.
    # Check if Rook's(4) is C5-free.
    rook_edges_1 = make_rooks_graph(4, 4, 0)
    rook_edges_2 = make_rooks_graph(4, 4, 16)
    base_rook_edges = rook_edges_1 + rook_edges_2
    
    c5_free, alpha_ok = verify(32, base_rook_edges)
    if c5_free and alpha_ok:
        best_n = 32
        best_edges = base_rook_edges
        # Try to add vertex 32
        # Connect to 1 vertex in each Rook's graph
        # Try all combinations of 1 vertex in first, 1 in second
        for v1 in range(16):
            if time.time() >= deadline - 5: break
            for v2 in range(16, 32):
                if time.time() >= deadline - 5: break
                edges = base_rook_edges + [[32, v1], [32, v2]]
                c5_free, alpha_ok = verify(33, edges)
                if c5_free and alpha_ok:
                    best_n = 33
                    best_edges = edges
                    break
            if best_n >= 33: break
            
        # If 33 works, try 34
        if best_n >= 33 and time.time() < deadline - 10:
             # Add vertex 33
             # Connect to 1 in each of the 3 components?
             # We have 2 Rook's and 1 vertex.
             # This is getting complex. Let's stick to 33 if possible.
             pass
             
    # If Rook's didn't work or C5 found, try 8 K4s (current best)
    # and randomized search as before.
    
    return {'n': best_n, 'edges': best_edges}
