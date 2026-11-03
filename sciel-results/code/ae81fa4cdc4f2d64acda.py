import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = nm * mp // m
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    def verify(U, V, W):
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-10)

    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        terms = []
        # M1 = (A11 + A22) * (B11 + B22)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M2 = A22 * (B21 - B11)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i1] = 1.0; w[k0*n+i1] = 1.0 # Wait, W index is k*n+i.
        # Correction: W[k0*n+i1] gets contribution from M2.
        # M2 contributes to C21 and C22?
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        # M2 = A22 (B21 - B11).
        # C21 += A22 B21. C22 -= A22 B11.
        # So W[k0*n+i1] += 1, W[k1*n+i1] -= 1? No.
        # B21 is at j1, k0. B11 is at j0, k0.
        # So v[j1*p+k0]=1, v[j0*p+k0]=-1.
        # u[i1*m+j1]=1.
        # Result is at i1, k0. So w[k0*n+i1]=1.
        # And at i1, k1? No, B11 and B21 are both in col k0.
        # So only w[k0*n+i1] is affected?
        # Wait, Strassen formulas:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1? No.
        # Standard Strassen:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = A22 * (B21 - B11)
        # M3 = (A11 + A12) * B22
        # M4 = A11 * (B12 - B22)
        # M5 = (A12 + A22) * (B11 + B12)
        # M6 = (A11 - A22) * (B21 + B22)
        # M7 = (A12 - A21) * (B11 + B21)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        
        # Let's re-verify the terms generated in the previous code.
        # The previous code had:
        # Term 2: u[i1*m+j0]=1, u[i1*m+j1]=1 ... this looks like M2 or similar?
        # Actually, the previous code's Strassen implementation was likely correct as it achieved rank 99.
        # I will keep the existing strassen_block_terms logic to avoid introducing bugs, 
        # but I will double check the W indices.
        
        # Let's rewrite strassen_block_terms carefully.
        pass

    # Re-implementing Strassen terms carefully
    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        terms = []
        
        # M1 = (A11 + A22) * (B11 + B22)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0 # C11
        w[k1*n+i0] = 1.0 # C12
        w[k0*n+i1] = 1.0 # C21
        w[k1*n+i1] = 1.0 # C22
        terms.append((u, v, w))
        
        # M2 = A22 * (B21 - B11)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i1] = 1.0 # C21
        terms.append((u, v, w))
        
        # M3 = (A11 + A12) * B22
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k1*n+i0] = 1.0 # C12
        terms.append((u, v, w))
        
        # M4 = A11 * (B12 - B22)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0 # C12
        w[k0*n+i0] = 1.0 # C11
        terms.append((u, v, w))
        
        # M5 = (A12 + A22) * (B11 + B12)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k0*n+i0] = -1.0 # C11
        w[k1*n+i0] = 1.0 # C12
        w[k0*n+i1] = 1.0 # C21
        w[k1*n+i1] = 1.0 # C22
        terms.append((u, v, w))
        
        # M6 = (A11 - A22) * (B21 + B22)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k1*n+i1] = 1.0 # C22
        terms.append((u, v, w))
        
        # M7 = (A12 - A21) * (B11 + B21)
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j1*p+k0] = 1.0
        w[k0*n+i0] = 1.0 # C11
        w[k0*n+i1] = 1.0 # C21
        terms.append((u, v, w))
        
        return terms

    def build_decomposition(blocks):
        U_list, V_list, W_list = [], [], []
        covered = set()
        
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
            for u, v, w in terms:
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1.0
                        v[j*p+k] = 1.0
                        w[k*n+i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return U, V, W

    def is_valid_block_set(blocks):
        seen = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in seen:
                            return False
                        seen.add((i, j, k))
        return True

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    I_sets = [frozenset({i, i+1}) for i in range(n-1)]
    J_sets = [frozenset({j, j+1}) for j in range(m-1)]
    K_sets = [frozenset({k, k+1}) for k in range(p-1)]
    
    all_blocks = []
    for I in I_sets:
        for J in J_sets:
            for K in K_sets:
                all_blocks.append((I, J, K))
                
    # Improved packing search
    def compatible(b1, b2):
        I1, J1, K1 = b1
        I2, J2, K2 = b2
        return not (I1 & I2 and J1 & J2 and K1 & K2)

    def greedy_packing_with_local_search(max_iter=200):
        best_count = 0
        best_set = []
        
        for _ in range(max_iter):
            current_set = []
            remaining = list(all_blocks)
            random.shuffle(remaining)
            
            # Greedy add
            for b in remaining:
                if all(compatible(b, cb) for cb in current_set):
                    current_set.append(b)
            
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set[:]
                
            # Local search: try to swap or add
            # Simplified: just restarts are often enough for small N
            if best_count >= 10: break # Unlikely, but safety
            
        return best_set

    if time.time() < deadline - 20:
        found_blocks = greedy_packing_with_local_search(100)
        if is_valid_block_set(found_blocks):
            U, V, W = build_decomposition(found_blocks)
            if verify(U, V, W):
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W

    # Optimization: Remove redundant terms
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
        R = U.shape[1]
        
        removed_any = True
        while removed_any and time.time() < deadline - 10:
            removed_any = False
            for c in range(R):
                if time.time() >= deadline - 10:
                    break
                # Check if term c is zero
                if not np.any(U[:, c]) or not np.any(V[:, c]) or not np.any(W[:, c]):
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
            # Check for linear dependency or zero contribution
            # This is expensive. Skip for now as Strassen terms are independent.
        
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
