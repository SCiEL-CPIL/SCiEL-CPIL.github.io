import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # The known maximum n for R(K2,2,2) > n is 30.
    # R(K2,2,2) is known to be 31, so n=30 is the max valid coloring size.
    # We try to find a 30-vertex coloring.
    
    n_target = 30
    
    # Fast verification for K2,2,2
    def has_k222(adj_list, n):
        # Check if color 0 (adj) contains K2,2,2
        # K2,2,2 exists if there is a non-edge (u, v) such that the induced subgraph
        # on N(u) intersection N(v) contains a K2,2.
        # K2,2 exists in a graph H if there are adjacent vertices x, y in H 
        # with at least 2 common neighbors in H.
        
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (u, v) is a non-edge in color 0
                if (adj_list[u] >> v) & 1:
                    continue
                cn_mask = adj_list[u] & adj_list[v]
                
                # Iterate over edges (x, y) in the induced subgraph on cn_mask
                m = cn_mask
                while m:
                    lsb_x = m & (-m)
                    x = lsb_x.bit_length() - 1
                    m ^= lsb_x
                    
                    # Neighbors of x in cn_mask
                    nx_mask = adj_list[x] & cn_mask
                    
                    # Iterate over neighbors y of x in cn_mask
                    my = nx_mask
                    while my:
                        lsb_y = my & (-my)
                        y = lsb_y.bit_length() - 1
                        my ^= lsb_y
                        
                        # Common neighbors of x and y in cn_mask
                        common = nx_mask & adj_list[y] & cn_mask
                        if common.bit_count() >= 2:
                            return True
        return False

    def get_colors_from_adj(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def adj_to_mask_list(adj_matrix, n):
        adj_list = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    adj_list[i] |= (1 << j)
                    adj_list[j] |= (1 << i)
        return adj_list

    # Spectral objective: sum of largest eigenvalues of A and A_bar
    def spectral_obj(adj_matrix, n):
        # A is symmetric 0-1 matrix
        # A_bar is J - I - A
        # We want to minimize lambda_1(A) + lambda_1(A_bar)
        
        # For speed, we can use power iteration or numpy's eigh
        # Since n=30, eigh is fast enough.
        
        # Ensure float for eigh
        A = adj_matrix.astype(float)
        J = np.ones((n, n))
        I = np.eye(n)
        A_bar = J - I - A
        
        # Largest eigenvalue of symmetric matrix is the max of eigh
        # eigh returns sorted eigenvalues
        eig_A = np.linalg.eigvalsh(A)
        eig_A_bar = np.linalg.eigvalsh(A_bar)
        
        return eig_A[-1] + eig_A_bar[-1]

    # Local search
    best_score = float('inf')
    best_adj = None
    
    # Initialize with Paley graph for n=29 padded to 30? No, random start is better for exploration.
    # Or start with a random graph.
    
    np.random.seed(42)
    
    # Start with a random graph
    adj = np.zeros((n_target, n_target), dtype=int)
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if np.random.rand() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
                
    current_score = spectral_obj(adj, n_target)
    best_score = current_score
    best_adj = adj.copy()
    
    # Local search loop
    while time.time() < deadline - 10.0:
        improved = False
        # Try flipping each edge once
        for i in range(n_target):
            for j in range(i + 1, n_target):
                # Flip edge (i, j)
                old_val = adj[i, j]
                adj[i, j] = 1 - old_val
                adj[j, i] = 1 - old_val
                
                new_score = spectral_obj(adj, n_target)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                else:
                    # Revert
                    adj[i, j] = old_val
                    adj[j, i] = old_val
        
        if not improved:
            # Perturb: flip a random set of edges to escape local minimum
            num_flips = 5
            for _ in range(num_flips):
                i = np.random.randint(n_target)
                j = np.random.randint(n_target)
                if i != j:
                    if i > j: i, j = j, i
                    adj[i, j] = 1 - adj[i, j]
                    adj[j, i] = 1 - adj[i, j]
            current_score = spectral_obj(adj, n_target)
            
    # Check if best_adj is valid
    # If spectral condition is met, it's likely valid, but we must verify.
    # A sufficient condition for K_2,2,2 free is lambda_1 <= 2.
    # If best_score <= 4.0, then lambda_1(A) <= 2 and lambda_1(A_bar) <= 2, so it's valid.
    
    adj_list = adj_to_mask_list(best_adj, n_target)
    if has_k222(adj_list, n_target):
        # Check complement
        comp_adj = [0] * n_target
        full = (1 << n_target) - 1
        for i in range(n_target):
            comp_adj[i] = full ^ adj_list[i] ^ (1 << i)
        if has_k222(comp_adj, n_target):
            # Not valid, fallback to n=29 Paley
            n = 29
            residues = set()
            for x in range(1, n):
                residues.add((x * x) % n)
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % n
                    if diff in residues:
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': n, 'colors': colors}
            
    # If valid, return n=30
    colors = get_colors_from_adj(best_adj, n_target)
    return {'n': n_target, 'colors': colors}
