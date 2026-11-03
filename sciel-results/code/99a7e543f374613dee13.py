import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        def term_sparsity(idx):
            u_zeros = sum(1 for i in range(n*m) if U_list[i][idx] == 0)
            v_zeros = sum(1 for j in range(m*p) if V_list[j][idx] == 0)
            w_zeros = sum(1 for k in range(p*n) if W_list[k][idx] == 0)
            return u_zeros + v_zeros + w_zeros
        
        for _ in range(20):
            if time_left() < 5:
                break
            progress = False
            order = sorted(range(rank), key=term_sparsity, reverse=True)
            for term_idx in order:
                if time_left() < 5:
                    break
                if term_idx >= rank:
                    continue
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
        
        max_pairs = min(2000, rank * (rank - 1) // 2)
        pairs_checked = 0
        
        for idx1 in range(rank):
            if time_left() < 10 or pairs_checked >= max_pairs:
                break
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10 or pairs_checked >= max_pairs:
                    break
                pairs_checked += 1
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][idx1] = 0
                    U_new[i][idx2] = 0
                for j in range(m*p):
                    V_new[j][idx1] = 0
                    V_new[j][idx2] = 0
                for k in range(p*n):
                    W_new[k][idx1] = 0
                    W_new[k][idx2] = 0
                    
                if verify(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once - more aggressive but still bounded."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
        
        max_triples = min(500, rank * (rank-1) * (rank-2) // 6)
        triples_checked = 0
        
        for idx1 in range(rank):
            if time_left() < 10 or triples_checked >= max_triples:
                break
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10 or triples_checked >= max_triples:
                    break
                for idx3 in range(idx2 + 1, rank):
                    if time_left() < 10 or triples_checked >= max_triples:
                        break
                    triples_checked += 1
                    
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][idx1] = 0
                        U_new[i][idx2] = 0
                        U_new[i][idx3] = 0
                    for j in range(m*p):
                        V_new[j][idx1] = 0
                        V_new[j][idx2] = 0
                        V_new[j][idx3] = 0
                    for k in range(p*n):
                        W_new[k][idx1] = 0
                        W_new[k][idx2] = 0
                        W_new[k][idx3] = 0
                        
                    if verify(U_new, V_new, W_new, rank):
                        keep = [j for j in range(rank) if j not in (idx1, idx2, idx3)]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 3
                        return try_triple_elimination(U_list, V_list, W_list, rank)
                        
        return U_list, V_list, W_list, rank

    def try_linear_combination_merge(U_list, V_list, W_list, rank):
        """
        Try to find two terms a, b such that their tensor vectors are proportional:
        vec_a = c * vec_b
        In that case, we can merge them into a single term.
        """
        if time_left() < 15:
            return U_list, V_list, W_list, rank
        
        # For each pair of terms, check if their tensor vectors are proportional
        # vec_t[i*m*p + j*p + k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Precompute tensor vectors
        total_dim = n * m * p
        vecs = []
        for t in range(rank):
            vec = np.zeros(total_dim, dtype=float)
            for i in range(n):
                for j in range(m):
                    u_val = U_list[i*m+j][t]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_list[j*p+k][t]
                        w_val = W_list[k*n+i][t]
                        if v_val == 0 or w_val == 0:
                            continue
                        vec[i*m*p + j*p + k] = u_val * v_val * w_val
            vecs.append(vec)
        
        # Check pairs for proportionality
        for a in range(rank):
            if time_left() < 10:
                break
            for b in range(a + 1, rank):
                if time_left() < 10:
                    break
                # Check if vec_a and vec_b are proportional
                # Find a non-zero component in vec_a
                nz = np.nonzero(vecs[a])[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                if abs(vecs[a][idx]) < 1e-12:
                    continue
                c = vecs[b][idx] / vecs[a][idx]
                
                # Check if vec_b ≈ c * vec_a
                if np.allclose(vecs[b], c * vecs[a], rtol=1e-8, atol=1e-10):
                    # They are proportional! Merge them.
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    # Scale term a by (1+c)
                    for i in range(n*m):
                        U_new[i][a] *= (1 + c)
                    
                    # Zero out term b
                    for i in range(n*m):
                        U_new[i][b] = 0
                    for j in range(m*p):
                        V_new[j][b] = 0
                    for k in range(p*n):
                        W_new[k][b] = 0
                    
                    # Remove term b
                    keep = [j for j in range(rank) if j != b]
                    U_list = [[row[j] for j in keep] for row in U_new]
                    V_list = [[row[j] for j in keep] for row in V_new]
                    W_list = [[row[j] for j in keep] for row in W_new]
                    rank -= 1
                    
                    # Verify
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list, rank
        
        return U_list, V_list, W_list, rank

    def try_structured_reduction(U_list, V_list, W_list, rank):
        """
        Apply a structured approach: identify blocks of terms that share
        similar sparsity patterns and try to reduce within those blocks.
        This mimics the group algebra structure where terms are organized
        by their support patterns.
        """
        if time_left() < 20:
            return U_list, V_list, W_list, rank
        
        # Group terms by their U-support pattern
        u_supports = {}
        for t in range(rank):
            support = tuple(i for i in range(n*m) if U_list[i][t] != 0)
            if support not in u_supports:
                u_supports[support] = []
            u_supports[support].append(t)
        
        # For each group with multiple terms, try to reduce
        for support, terms in u_supports.items():
            if time_left() < 10:
                break
            if len(terms) < 2:
                continue
            
            # Try to eliminate terms within this group
            for t in terms:
                if time_left() < 5:
                    break
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][t] = 0
                for j in range(m*p):
                    V_new[j][t] = 0
                for k in range(p*n):
                    W_new[k][t] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    U_list = [[row[j] for j in range(rank) if j != t] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != t] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != t] for row in W_list]
                    rank -= 1
        
        return U_list, V_list, W_list, rank

    # Try different configurations - more comprehensive search
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(3,5),(2,4)]),
        ([(0,1),(2,3)], [(1,2),(3,4),(0,5)]),
        ([(0,1),(2,3)], [(0,2),(3,5),(1,4)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try linear combination merge if time permits
        if time_left() > 50:
            U_list, V_list, W_list, new_rank = try_linear_combination_merge(U_list, V_list, W_list, new_rank)
        
        # Try structured reduction
        if time_left() > 40:
            U_list, V_list, W_list, new_rank = try_structured_reduction(U_list, V_list, W_list, new_rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 30:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
        
        # Try triple elimination if time permits
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_triple_elimination(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
