import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 125

    def time_left():
        return deadline - time.time()

    def to_frac_list(mat):
        return [[Fraction(int(x), 1) for x in row] for row in mat]

    def verify_exact(U, V, W, rank):
        """
        Verify decomposition using exact arithmetic.
        U, V, W are lists of lists of Fractions.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition with integer coefficients."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(remaining_j) * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0][col] += 1; U[1*m+j1][col] += 1
                V[j0*p+k0][col] += 1; V[j1*p+k1][col] += 1
                W[k0*n+0][col] += 1; W[k1*n+1][col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0][col] += 1; U[1*m+j1][col] += 1
                V[j0*p+k0][col] += 1
                W[k0*n+1][col] += 1; W[k1*n+1][col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0][col] += 1
                V[j0*p+k1][col] += 1; V[j1*p+k1][col] -= 1
                W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1][col] += 1
                V[j1*p+k0][col] += 1; V[j0*p+k0][col] -= 1
                W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0][col] += 1; U[0*m+j1][col] += 1
                V[j1*p+k1][col] += 1
                W[k0*n+0][col] -= 1; W[k1*n+0][col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0][col] += 1; U[0*m+j0][col] -= 1
                V[j0*p+k0][col] += 1; V[j0*p+k1][col] += 1
                W[k1*n+1][col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1][col] += 1; U[1*m+j1][col] -= 1
                V[j1*p+k0][col] += 1; V[j1*p+k1][col] += 1
                W[k0*n+0][col] += 1
                col += 1
        
        for j in remaining_j:
            for k in range(p):
                U[0*m+j][col] += 1
                V[j*p+k][col] += 1
                W[k*n+0][col] += 1
                col += 1
                U[1*m+j][col] += 1
                V[j*p+k][col] += 1
                W[k*n+1][col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms_exact(U, V, W, rank):
        """
        Try to eliminate terms using exact arithmetic.
        """
        U_f = to_frac_list(U)
        V_f = to_frac_list(V)
        W_f = to_frac_list(W)
        
        for _ in range(10):
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Check if term is redundant
                U_new = [row[:] for row in U_f]
                V_new = [row[:] for row in V_f]
                W_new = [row[:] for row in W_f]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify_exact(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_f = [[row[j] for j in range(rank) if j != term_idx] for row in U_f]
                    V_f = [[row[j] for j in range(rank) if j != term_idx] for row in V_f]
                    W_f = [[row[j] for j in range(rank) if j != term_idx] for row in W_f]
                    rank -= 1
                    progress = True
            if not progress:
                break

        # Convert back to float/int for return
        U_out = [[float(x) for x in row] for row in U_f]
        V_out = [[float(x) for x in row] for row in V_f]
        W_out = [[float(x) for x in row] for row in W_f]
        
        return U_out, V_out, W_out, rank

    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        # Verify initial construction
        if not verify_exact(to_frac_list(U), to_frac_list(V), to_frac_list(W), rank):
            continue
        
        U_list, V_list, W_list, new_rank = try_eliminate_terms_exact(U, V, W, rank)
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
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
