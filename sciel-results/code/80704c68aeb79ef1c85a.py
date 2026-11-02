import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # Safety margin

    def verify_decomp(U, V, W, trials=30):
        """Fast verification using numpy vectorized ops."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for _ in range(trials):
            A = np.random.randint(-10, 11, size=(n, m))
            B = np.random.randint(-10, 11, size=(m, p))
            C_true = A @ B
            u_vals = (Ua * A.flatten()[:, None]).sum(axis=0)
            v_vals = (Va * B.flatten()[:, None]).sum(axis=0)
            W_r = Wa.reshape(p, n, rank)
            C_calc = (W_r * (u_vals * v_vals)[None, None, :]).sum(axis=2).T
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7."""
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return U, V, W

    def try_reduce_rank(U, V, W, max_reduce=2):
        """
        Try to reduce the rank by finding linear dependencies among the rank-1 terms.
        Each term t contributes: outer(U[:,t], V[:,t]) * W[:,t]
        If we can express one term as a linear combination of others, we can eliminate it.
        
        We search for subsets of size (rank - r) that still span the tensor space.
        """
        Ua = U.copy()
        Va = V.copy()
        Wa = W.copy()
        rank = Ua.shape[1]
        
        # For each candidate removal, check if the remaining terms still work
        # This is expensive, so we limit the search
        for r in range(1, max_reduce + 1):
            if time.time() > deadline - 10:
                break
            # Try removing r terms at a time
            # For r=1: try removing each term one at a time
            if r == 1:
                for t in range(rank):
                    if time.time() > deadline - 5:
                        break
                    # Check if term t is redundant
                    # Build tensor from all terms except t
                    U_reduced = np.delete(Ua, t, axis=1)
                    V_reduced = np.delete(Va, t, axis=1)
                    W_reduced = np.delete(Wa, t, axis=1)
                    new_rank = U_reduced.shape[1]
                    
                    # Quick check: verify with a few random inputs
                    valid = True
                    for _ in range(10):
                        A = np.random.randint(-5, 6, size=(n, m))
                        B = np.random.randint(-5, 6, size=(m, p))
                        C_true = A @ B
                        u_vals = (U_reduced * A.flatten()[:, None]).sum(axis=0)
                        v_vals = (V_reduced * B.flatten()[:, None]).sum(axis=0)
                        W_r = W_reduced.reshape(p, n, new_rank)
                        C_calc = (W_r * (u_vals * v_vals)[None, None, :]).sum(axis=2).T
                        if not np.allclose(C_true, C_calc, atol=1e-8):
                            valid = False
                            break
                    
                    if valid:
                        # Found a reduction! Verify more thoroughly
                        if verify_decomp(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), trials=20):
                            return U_reduced, V_reduced, W_reduced
            elif r == 2:
                # Try removing pairs of terms
                for t1 in range(rank):
                    if time.time() > deadline - 5:
                        break
                    for t2 in range(t1 + 1, rank):
                        if time.time() > deadline - 5:
                            break
                        # Build tensor from all terms except t1 and t2
                        mask = [i for i in range(rank) if i != t1 and i != t2]
                        U_reduced = Ua[:, mask]
                        V_reduced = Va[:, mask]
                        W_reduced = Wa[:, mask]
                        new_rank = U_reduced.shape[1]
                        
                        # Quick check
                        valid = True
                        for _ in range(8):
                            A = np.random.randint(-5, 6, size=(n, m))
                            B = np.random.randint(-5, 6, size=(m, p))
                            C_true = A @ B
                            u_vals = (U_reduced * A.flatten()[:, None]).sum(axis=0)
                            v_vals = (V_reduced * B.flatten()[:, None]).sum(axis=0)
                            W_r = W_reduced.reshape(p, n, new_rank)
                            C_calc = (W_r * (u_vals * v_vals)[None, None, :]).sum(axis=2).T
                            if not np.allclose(C_true, C_calc, atol=1e-8):
                                valid = False
                                break
                        
                        if valid:
                            if verify_decomp(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), trials=20):
                                return U_reduced, V_reduced, W_reduced
        
        return Ua, Va, Wa

    def build_dft_rank49():
        """
        DFT-based decomposition placeholder.
        """
        return None

    # Build the rank-25 decomposition
    U, V, W = build_rank25()
    
    # Try to reduce the rank via global algebraic optimization
    U_opt, V_opt, W_opt = try_reduce_rank(U, V, W, max_reduce=2)
    
    # Verify the optimized decomposition
    if verify_decomp(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # If optimization failed or produced invalid result, use original
    if verify_decomp(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
