import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Build the rank-39 decomposition
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
            
            # m1 = (a+d)(e+h)
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m2 = (c+d)e
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            
            # m3 = a(f-h)
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m4 = d(g-e)
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            
            # m5 = (a+b)h
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            
            # m6 = (c-a)(e+f)
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            
            # m7 = (b-d)(g+h)
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Single column k2 = 10 using schoolbook
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
    rank = len(U[0])
    
    # Verify function
    def verify(U, V, W):
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        # Check that non-tensor entries are zero
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    is_tensor = False
                    for (tu, tv, tw) in tensor_entries:
                        if tu == u_idx and tv == v_idx and tw == w_idx:
                            is_tensor = True
                            break
                    if is_tensor:
                        continue
                    total = 0
                    for r in range(rank):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(total) > 1e-9:
                        return False
        return True

    # Try to reduce rank by removing one column and solving for W
    # We use Fraction for exact arithmetic
    def try_reduce_rank(U, V, W, target_rank):
        orig_rank = len(U[0])
        # Try removing each column
        for r_to_remove in range(orig_rank):
            if time.time() > deadline - 5:
                return None
            
            # Create U_rem, V_rem
            mask = [i for i in range(orig_rank) if i != r_to_remove]
            U_rem = [[row[i] for i in mask] for row in U]
            V_rem = [[row[i] for i in mask] for row in V]
            
            # Convert to Fraction
            U_frac = [[Fraction(x) for x in row] for row in U_rem]
            V_frac = [[Fraction(x) for x in row] for row in V_rem]
            
            W_new = [[0] * target_rank for _ in range(p * n)]
            success = True
            
            for w_idx in range(p * n):
                # Build matrix M (size (n*m * m*p) x target_rank)
                # and vector t (size (n*m * m*p))
                # M[row, r] = U_rem[u, r] * V_rem[v, r]
                # t[row] = 1 if (u, v, w_idx) is a tensor entry, else 0
                
                M = []
                t = []
                for u_idx in range(n * m):
                    for v_idx in range(m * p):
                        row_M = []
                        for r in range(target_rank):
                            row_M.append(U_frac[u_idx][r] * V_frac[v_idx][r])
                        M.append(row_M)
                        
                        # Check if (u_idx, v_idx, w_idx) is a tensor entry
                        is_tensor = False
                        for (tu, tv, tw) in tensor_entries:
                            if tu == u_idx and tv == v_idx and tw == w_idx:
                                is_tensor = True
                                break
                        t.append(Fraction(1) if is_tensor else Fraction(0))
                
                # Solve M * x = t
                # Use Gaussian elimination with Fractions
                # M is 88 x 38
                # We can use numpy for speed if we convert to float, but we need exactness.
                # 88x38 is small enough for Fraction Gaussian elimination?
                # 88*38*88 operations ~ 300k. With Fractions, it might be slow.
                # Let's try numpy first to check consistency, then refine?
                # No, just use numpy lstsq and check if residual is small.
                # If residual is small, we assume it's consistent and the solution is rational.
                
                M_np = np.array(M, dtype=float)
                t_np = np.array(t, dtype=float)
                
                try:
                    x_np, residuals, rank_np, s = np.linalg.lstsq(M_np, t_np, rcond=None)
                    # Check residual
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            success = False
                            break
                    else:
                        # Check if M is full rank and consistent
                        # Actually, if rank_np < target_rank, it might still be consistent.
                        # Let's check the norm of M*x - t
                        residual = np.linalg.norm(M_np @ x_np - t_np)
                        if residual > 1e-9:
                            success = False
                            break
                except np.linalg.LinAlgError:
                    success = False
                    break
                
                # If consistent, convert x_np to Fraction
                # We assume the solution is rational.
                # To get the exact Fraction, we can solve using Fractions.
                # But doing this for every w_idx and every r_to_remove might be slow.
                # Let's only do the Fraction solve if the numpy check passes.
                
                # Solve M x = t exactly
                # We can use a simple Gaussian elimination function for Fractions
                def solve_exact(A, b):
                    n = len(A)
                    m = len(A[0])
                    # Augmented matrix
                    M_aug = [row[:] + [b[i]] for i, row in enumerate(A)]
                    
                    for col in range(m):
                        # Find pivot
                        pivot_row = -1
                        for row in range(col, n):
                            if M_aug[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            continue # Free variable or inconsistent?
                        
                        # Swap rows
                        M_aug[col], M_aug[pivot_row] = M_aug[pivot_row], M_aug[col]
                        
                        # Normalize pivot row
                        pivot_val = M_aug[col][col]
                        for j in range(col, m + 1):
                            M_aug[col][j] /= pivot_val
                            
                        # Eliminate other rows
                        for row in range(n):
                            if row != col:
                                factor = M_aug[row][col]
                                if factor != 0:
                                    for j in range(col, m + 1):
                                        M_aug[row][j] -= factor * M_aug[col][j]
                    
                    # Check for inconsistency
                    for i in range(n):
                        # If row is 0...0 = non-zero
                        row_sum = 0
                        for j in range(m):
                            row_sum += M_aug[i][j]
                        if row_sum == 0 and M_aug[i][m] != 0:
                            return None # Inconsistent
                    
                    # Extract solution
                    # If underdetermined, we pick one solution (e.g., set free vars to 0)
                    x = [Fraction(0)] * m
                    for col in range(m):
                        if M_aug[col][col] == 1:
                            x[col] = M_aug[col][m]
                        else:
                            # This col is a free variable or dependent
                            # In our case, if it's consistent, we just need *a* solution.
                            # If M is full rank, this won't happen.
                            pass
                    return x

                x_exact = solve_exact(M, t)
                if x_exact is None:
                    success = False
                    break
                
                for r in range(target_rank):
                    W_new[w_idx][r] = x_exact[r]
            
            if success:
                # Check if entries are integers or half-integers
                valid = True
                for row in W_new:
                    for val in row:
                        # Check if val is integer or half-integer
                        # val is a Fraction
                        if val.denominator not in [1, 2]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert back to int/float for return
                    U_final = [[int(x) for x in row] for row in U_rem]
                    V_final = [[int(x) for x in row] for row in V_rem]
                    W_final = []
                    for row in W_new:
                        W_final.append([float(x) if x.denominator == 2 else int(x) for x in row])
                    
                    # Verify
                    if verify(U_final, V_final, W_final):
                        return U_final, V_final, W_final
                    else:
                        # Verification failed, maybe due to float conversion?
                        # Try returning Fractions? The interface expects numeric types.
                        # Let's try to keep them as floats if they are exact halves.
                        pass
                        return None
            
    # Try to reduce to rank 38
    reduced = try_reduce_rank(U, V, W, 38)
    if reduced:
        U, V, W = reduced
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 39
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
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
