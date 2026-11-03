import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

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
        # Each 2x2x2 block: rank 7
        # Remaining j: standard 2*p terms per j
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
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

    def build_2x2x3_decomp(j0, j1, k0, k1, k2):
        """Build 2x2x3 decomposition: Rank 11 (7 from 2x2x2 + 4 from 2x2x1)."""
        rank = 11
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        # 2x2x2 part for k0, k1
        # P1
        U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
        col += 1
        # P2
        U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
        col += 1
        # P3
        U[0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
        col += 1
        # P4
        U[1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
        col += 1
        # P5
        U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
        col += 1
        # P6
        U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+1, col] += 1
        col += 1
        # P7
        U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+0, col] += 1
        col += 1
        
        # 2x2x1 part for k2
        # A0j0 * Bj0k2
        U[0*m+j0, col] += 1
        V[j0*p+k2, col] += 1
        W[k2*n+0, col] += 1
        col += 1
        # A1j0 * Bj0k2
        U[1*m+j0, col] += 1
        V[j0*p+k2, col] += 1
        W[k2*n+1, col] += 1
        col += 1
        # A0j1 * Bj1k2
        U[0*m+j1, col] += 1
        V[j1*p+k2, col] += 1
        W[k2*n+0, col] += 1
        col += 1
        # A1j1 * Bj1k2
        U[1*m+j1, col] += 1
        V[j1*p+k2, col] += 1
        W[k2*n+1, col] += 1
        col += 1
        
        return U, V, W, rank

    def build_2x3x3_decomp(j0, j1, j2, k0, k1, k2):
        """Build 2x3x3 decomposition: Rank 17 (11 from 2x2x3 + 6 from 2x1x3)."""
        rank = 17
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        # 2x2x3 part for j0, j1
        U_sub, V_sub, W_sub, rank_sub = build_2x2x3_decomp(j0, j1, k0, k1, k2)
        U[:, col:col+rank_sub] += U_sub
        V[:, col:col+rank_sub] += V_sub
        W[:, col:col+rank_sub] += W_sub
        col += rank_sub
        
        # 2x1x3 part for j2
        for k in [k0, k1, k2]:
            # A0j2 * Bj2k
            U[0*m+j2, col] += 1
            V[j2*p+k, col] += 1
            W[k*n+0, col] += 1
            col += 1
            # A1j2 * Bj2k
            U[1*m+j2, col] += 1
            V[j2*p+k, col] += 1
            W[k*n+1, col] += 1
            col += 1
            
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for term_idx in range(rank - 1, -1, -1):
            if time_left() < 5:
                break
            # Check if term is redundant
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
                # Remove the column
                U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                rank -= 1
        
        return U_list, V_list, W_list, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    # Standard Strassen configurations
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            
    # Mixed block configurations
    # Config: m=3+2, p=3+3
    # Block 1: 2x3x3 (j:0,1,2; k:0,1,2)
    # Block 2: 2x3x3 (j:0,1,2; k:3,4,5)
    # Block 3: 2x2x3 (j:3,4; k:0,1,2)
    # Block 4: 2x2x3 (j:3,4; k:3,4,5)
    
    if time_left() > 20:
        U1, V1, W1, r1 = build_2x3x3_decomp(0, 1, 2, 0, 1, 2)
        U2, V2, W2, r2 = build_2x3x3_decomp(0, 1, 2, 3, 4, 5)
        U3, V3, W3, r3 = build_2x2x3_decomp(3, 4, 0, 1, 2)
        U4, V4, W4, r4 = build_2x2x3_decomp(3, 4, 3, 4, 5)
        
        U_total = np.hstack([U1, U2, U3, U4])
        V_total = np.hstack([V1, V2, V3, V4])
        W_total = np.hstack([W1, W2, W3, W4])
        
        if verify(U_total.tolist(), V_total.tolist(), W_total.tolist(), r1+r2+r3+r4):
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U_total, V_total, W_total, r1+r2+r3+r4)
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
