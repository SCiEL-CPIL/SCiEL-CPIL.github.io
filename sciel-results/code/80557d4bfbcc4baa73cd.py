import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Build tensor entries
    tensor_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.add((u_idx, v_idx, w_idx))

    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            # m1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m2
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            
            # m3
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m4
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            
            # m5
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            
            # m6
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            
            # m7
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return U, V, W

    U, V, W = make_rank39()
    
    def verify(U, V, W):
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        
        # Check non-tensor entries are zero
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) in tensor_entries:
                        continue
                    total = 0
                    for r in range(rank):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(total) > 1e-9:
                        return False
        return True

    def try_reduce_rank(U, V, W, target_rank):
        orig_rank = len(U[0])
        for r_to_remove in range(orig_rank):
            if time.time() > deadline - 5:
                return None
            
            mask = [i for i in range(orig_rank) if i != r_to_remove]
            U_rem = np.array([[row[i] for i in mask] for row in U], dtype=float)
            V_rem = np.array([[row[i] for i in mask] for row in V], dtype=float)
            
            W_new = np.zeros((p * n, target_rank), dtype=float)
            success = True
            
            for w_idx in range(p * n):
                # Build M
                M = np.zeros((n * m * m * p, target_rank))
                t = np.zeros(n * m * m * p)
                
                idx = 0
                for u_idx in range(n * m):
                    for v_idx in range(m * p):
                        for r in range(target_rank):
                            M[idx, r] = U_rem[u_idx, r] * V_rem[v_idx, r]
                        if (u_idx, v_idx, w_idx) in tensor_entries:
                            t[idx] = 1
                        idx += 1
                
                # Solve M x = t
                try:
                    x, residuals, rank_np, s = np.linalg.lstsq(M, t, rcond=None)
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            success = False
                            break
                    else:
                        residual = np.linalg.norm(M @ x - t)
                        if residual > 1e-9:
                            success = False
                            break
                except np.linalg.LinAlgError:
                    success = False
                    break
                
                W_new[w_idx] = x
            
            if success:
                # Check if entries are close to half-integers
                valid = True
                W_frac = []
                for row in W_new:
                    frac_row = []
                    for val in row:
                        # Check if val is close to k/2
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_half) > 1e-6:
                            valid = False
                            break
                        frac_row.append(Fraction(nearest_half).limit_denominator(100))
                    if not valid:
                        break
                    W_frac.append(frac_row)
                
                if valid:
                    # Construct final matrices
                    U_final = [[int(x) for x in row] for row in U_rem]
                    V_final = [[int(x) for x in row] for row in V_rem]
                    W_final = []
                    for row in W_frac:
                        W_final.append([float(x) if x.denominator == 2 else int(x) for x in row])
                    
                    if verify(U_final, V_final, W_final):
                        return U_final, V_final, W_final
        
        return None

    reduced = try_reduce_rank(U, V, W, 38)
    if reduced:
        U, V, W = reduced
        return {"U": U, "V": V, "W": W}
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
