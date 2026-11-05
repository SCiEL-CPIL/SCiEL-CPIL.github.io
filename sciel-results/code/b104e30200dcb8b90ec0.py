"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 9
    dim_B = m * p  # 15
    dim_C = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Build target tensor
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook solution: rank = n*m*p = 45
    rank_sb = n * m * p
    U_sb = np.zeros((dim_A, rank_sb))
    V_sb = np.zeros((dim_B, rank_sb))
    W_sb = np.zeros((dim_C, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rec = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(rec, 1.0, atol=1e-9)
    
    def compute_rec(U, V, W):
        rec = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return rec
    
    # Try to find better decompositions using Strassen-like ideas
    # Strategy: For each pair of k values, try to combine them
    # Also try combining across i and j dimensions
    
    # Generate candidate rank-1 tensors
    # Each is (u_vec, v_vec, w_vec)
    candidates = []
    
    # Add schoolbook as base
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(dim_A)
                v = np.zeros(dim_B)
                w = np.zeros(dim_C)
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u[a] = 1
                v[b] = 1
                w[c] = 1
                candidates.append((u, v, w))
    
    # Strassen-like: combine pairs of k for each (i,j)
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            u = np.zeros(dim_A)
                            v = np.zeros(dim_B)
                            w = np.zeros(dim_C)
                            a = i * m + j
                            b1 = j * p + k1
                            b2 = j * p + k2
                            c1 = k1 * n + i
                            c2 = k2 * n + i
                            u[a] = 1
                            v[b1] = s1
                            v[b2] = s2
                            w[c1] = 1
                            w[c2] = 1
                            candidates.append((u, v, w))
    
    # Combine pairs of i for each (j,k)
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            u = np.zeros(dim_A)
                            v = np.zeros(dim_B)
                            w = np.zeros(dim_C)
                            a1 = i1 * m + j
                            a2 = i2 * m + j
                            b = j * p + k
                            c1 = k * n + i1
                            c2 = k * n + i2
                            u[a1] = s1
                            u[a2] = s2
                            v[b] = 1
                            w[c1] = 1
                            w[c2] = 1
                            candidates.append((u, v, w))
    
    # Combine pairs of j for each (i,k)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            u = np.zeros(dim_A)
                            v = np.zeros(dim_B)
                            w = np.zeros(dim_C)
                            a1 = i * m + j1
                            a2 = i * m + j2
                            b1 = j1 * p + k
                            b2 = j2 * p + k
                            c = k * n + i
                            u[a1] = s1
                            u[a2] = s2
                            v[b1] = 1
                            v[b2] = 1
                            w[c] = 1
                            candidates.append((u, v, w))
    
    # Triple combinations: Strassen for 3x3 block
    # For 3x3 x 3x3, Strassen uses 19 mults. We can try similar here.
    # Generalized Strassen for combining multiple dimensions
    
    # Try 3-way combinations for k dimension
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    for k3 in range(k2+1, p):
                        # Try (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1) patterns
                        for signs in [
                            [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1],
                            [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
                        ]:
                            u = np.zeros(dim_A)
                            v = np.zeros(dim_B)
                            w = np.zeros(dim_C)
                            a = i * m + j
                            for kk, ss in zip([k1,k2,k3], signs):
                                b = j * p + kk
                                c = kk * n + i
                                v[b] = ss
                                w[c] = ss
                            u[a] = 1
                            candidates.append((u, v, w))
    
    # Now use greedy set cover to find minimal covering
    required_entries = []
    for a in range(dim_A):
        for b in range(dim_B):
            for c in range(dim_C):
                if T[a, b, c] == 1:
                    required_entries.append((a, b, c))
    
    # For each candidate, determine which required entries it covers
    candidate_coverage = []
    for u, v, w in candidates:
        covered = set()
        for (a, b, c) in required_entries:
            if u[a] * v[b] * w[c] != 0:
                covered.add((a, b, c))
        if covered:
            candidate_coverage.append(covered)
    
    # Greedy set cover
    uncovered = set(required_entries)
    selected_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = 0
        for idx, covered in enumerate(candidate_coverage):
            if idx in selected_indices:
                continue
            count = len(covered & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered -= candidate_coverage[best_idx]
    
    # Build the solution from selected candidates
    if selected_indices:
        R = len(selected_indices)
        U = np.zeros((dim_A, R))
        V = np.zeros((dim_B, R))
        W = np.zeros((dim_C, R))
        
        for r, idx in enumerate(selected_indices):
            u, v, w = candidates[idx]
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        
        # Verify and if valid, update best
        if verify(U, V, W):
            best_U = U
            best_V = V
            best_W = W
            best_rank = R
    
    # Local search: try removing columns and adjusting values
    if best_rank <= rank_sb and time.time() < deadline - 20:
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        # Try removing one column at a time
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for remove_col in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                # Try removing column remove_col
                if remove_col == 0:
                    test_U = current_U[:, 1:]
                    test_V = current_V[:, 1:]
                    test_W = current_W[:, 1:]
                elif remove_col == current_rank - 1:
                    test_U = current_U[:, :-1]
                    test_V = current_V[:, :-1]
                    test_W = current_W[:, :-1]
                else:
                    test_U = np.column_stack([current_U[:, :remove_col], current_U[:, remove_col+1:]])
                    test_V = np.column_stack([current_V[:, :remove_col], current_V[:, remove_col+1:]])
                    test_W = np.column_stack([current_W[:, :remove_col], current_W[:, remove_col+1:]])
                
                if verify(test_U, test_V, test_W):
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    current_rank -= 1
                    improved = True
                    if current_rank < best_rank:
                        best_U = current_U
                        best_V = current_V
                        best_W = current_W
                        best_rank = current_rank
                    break  # restart removal process
    
    # Try to reduce rank by combining columns with similar patterns
    if best_rank > 20 and time.time() < deadline - 30:
        # Try pairwise column elimination with value adjustment
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            found = False
            for r1 in range(current_rank):
                if time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, current_rank):
                    # Check if we can eliminate one of these two columns
                    # by adjusting the other
                    # Try eliminating r2
                    # Compute what r2 contributes
                    contrib_r2 = np.einsum('ij,jk,ki->ijk', 
                                           current_U[:, r2].reshape(n, m),
                                           current_V[:, r2].reshape(m, p),
                                           current_W[:, r2].reshape(p, n))
                    
                    # Try to absorb r2's contribution into r1
                    # This is complex, so just try simple cases
                    if np.allclose(current_U[:, r2], current_U[:, r1], atol=1e-9) and \
                       np.allclose(current_V[:, r2], current_V[:, r1], atol=1e-9) and \
                       np.allclose(current_W[:, r2], current_W[:, r1], atol=1e-9):
                        # Identical columns, can remove one
                        if r2 == current_rank - 1:
                            test_U = current_U[:, :r2]
                            test_V = current_V[:, :r2]
                            test_W = current_W[:, :r2]
                        else:
                            test_U = np.column_stack([current_U[:, :r2], current_U[:, r2+1:]])
                            test_V = np.column_stack([current_V[:, :r2], current_V[:, r2+1:]])
                            test_W = np.column_stack([current_W[:, :r2], current_W[:, r2+1:]])
                        
                        if verify(test_U, test_V, test_W):
                            current_U = test_U
                            current_V = test_V
                            current_W = test_W
                            current_rank -= 1
                            if current_rank < best_rank:
                                best_U = current_U
                                best_V = current_V
                                best_W = current_W
                                best_rank = current_rank
                            found = True
                            break
                if found:
                    break
    
    # Convert to lists of lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
