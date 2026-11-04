"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank
    
    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        """
        Return 7 rank-1 terms for a 2x2x2 Strassen block.
        Indices: i0<i1, j0<j1, k0<k1
        """
        terms = []
        
        # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M2: (A[i1,j0]+A[i1,j1])B[j0,k0]
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0
        w[k0*n+i1] = 1.0; w[k1*n+i1] = -1.0
        terms.append((u, v, w))
        
        # M3: A[i0,j0](B[j0,k1]-B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M4: A[i1,j1](B[j1,k0]-B[j0,k0])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i0] = 1.0; w[k0*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M5: (A[i0,j0]+A[i0,j1])B[j1,k1]
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k0*n+i0] = -1.0; w[k1*n+i0] = 1.0
        terms.append((u, v, w))
        
        # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i0*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0
        terms.append((u, v, w))
        
        return terms
    
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            block = (i0, i1, j0, j1, k0, k1)
                            # Compute the set of (i,j,k) tuples covered
                            covered = set()
                            for i in (i0, i1):
                                for j in (j0, j1):
                                    for k in (k0, k1):
                                        covered.add((i, j, k))
                            all_blocks.append((block, covered))
    
    # Find maximum set of disjoint blocks using greedy with multiple strategies
    import random
    random.seed(12345)
    
    best_blocks = []
    best_count = 0
    
    # Strategy 1: Greedy with random orderings
    for trial in range(500):
        if time.time() > deadline - 30:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        chosen = []
        used = set()
        for block, covered in shuffled:
            if not any(t in used for t in covered):
                chosen.append(block)
                used.update(covered)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]
    
    # Strategy 2: Greedy preferring blocks with "central" indices
    def block_priority(block):
        i0, i1, j0, j1, k0, k1 = block
        ci = (i0 + i1) / 2.0
        cj = (j0 + j1) / 2.0
        ck = (k0 + k1) / 2.0
        # Prefer blocks closer to center
        return -(abs(ci - (n-1)/2.0) + abs(cj - (m-1)/2.0) + abs(ck - (p-1)/2.0))
    
    sorted_blocks = sorted(all_blocks, key=lambda x: block_priority(x[0]), reverse=True)
    chosen = []
    used = set()
    for block, covered in sorted_blocks:
        if not any(t in used for t in covered):
            chosen.append(block)
            used.update(covered)
    if len(chosen) > best_count:
        best_count = len(chosen)
        best_blocks = chosen[:]
    
    # Strategy 3: Try to improve by local search - swap blocks
    # For now, use the best found
    print(f"Found {best_count} Strassen blocks")
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    covered = set()
    
    for block in best_blocks:
        i0, i1, j0, j1, k0, k1 = block
        terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
        for u, v, w in terms:
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    rank = len(U_list)
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    if not verify(U_mat, V_mat, W_mat):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, _ = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Represent each term as a vector in R^(nm*mp*pn)
    dim = nm * mp * pn
    if rank > 0:
        terms_vec = np.zeros((rank, dim))
        for c in range(rank):
            u = U_mat[:, c]
            v = V_mat[:, c]
            w = W_mat[:, c]
            terms_vec[c] = np.einsum('i,j,k->ijk', u, v, w).flatten()
        
        # The target tensor
        target = np.zeros((n, m, p))
        target[:] = 1.0
        target_flat = target.flatten()
        
        # Check if we can eliminate any term
        # For each term c, check if the remaining terms can still sum to target
        # This is: is target - term_c in the span of {term_j : j != c}?
        # But we need exact equality, not just span membership.
        # Actually, we need: sum_{j != c} alpha_j * term_j = target - term_c for some alphas
        # But the alphas must be such that the decomposition is valid.
        # This is complex. Instead, try a simpler check:
        # Can we set one coefficient to 0 and adjust others?
        
        # Simpler approach: check if any two terms are linearly dependent
        # If term_a = term_b, we can remove one
        for a in range(rank):
            if time.time() > deadline - 10:
                break
            for b in range(a+1, rank):
                if np.allclose(terms_vec[a], terms_vec[b], atol=1e-10):
                    # Terms a and b are identical, remove b
                    U_mat = np.delete(U_mat, b, axis=1)
                    V_mat = np.delete(V_mat, b, axis=1)
                    W_mat = np.delete(W_mat, b, axis=1)
                    rank -= 1
                    # Rebuild terms_vec
                    terms_vec = np.zeros((rank, dim))
                    for c in range(rank):
                        u = U_mat[:, c]
                        v = V_mat[:, c]
                        w = W_mat[:, c]
                        terms_vec[c] = np.einsum('i,j,k->ijk', u, v, w).flatten()
                    break
            if time.time() > deadline - 10:
                break
    
    # Final verification
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        U_sb, V_sb, W_sb, _ = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
