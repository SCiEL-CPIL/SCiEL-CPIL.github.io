import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def check_valid_fast(U_np, V_np, W_np, tol=1e-9):
        """Fast vectorized validity check."""
        r = U_np.shape[1]
        # Compute all triple products
        # U_np: (16, r), V_np: (28, r), W_np: (28, r)
        # For each (i,j,k): sum_c U[i*4+j,c] * V[j*7+k,c] * W[k*4+i,c]
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        # Remove zero columns
        nonzero_cols = [r for r in range(49) if any(U4[i][r] != 0 or V4[j][r] != 0 or W4[k][r] != 0 for i in range(16) for j in range(16) for k in range(16))]
        
        if len(nonzero_cols) < 49:
            U4 = [[U4[i][r] for r in nonzero_cols] for i in range(16)]
            V4 = [[V4[j][r] for r in nonzero_cols] for j in range(16)]
            W4 = [[W4[k][r] for r in nonzero_cols] for k in range(16)]
        
        return U4, V4, W4

    def build_4x4x7_decomposition():
        """
        Build a decomposition for 4x4x7 by combining the 4x4x4 decomposition with additional terms.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W

    def build_4x4x7_optimized():
        """
        Build 4x4x7 with a more optimized approach.
        Try to find shared structure between the 4x4x4 and 4x4x3 blocks.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        # Start with the 4x4x4 part
        U = [[0] * (r4 + 48) for _ in range(16)]
        V = [[0] * (r4 + 48) for _ in range(28)]
        W = [[0] * (r4 + 48) for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # For the 4x4x3 part, use schoolbook
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W

    def reduce_rank_greedy(U, V, W, max_iterations=2000):
        """
        Greedy rank reduction: try to eliminate columns by finding linear dependencies.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        
        best_U, best_V, best_W = U_np, V_np, W_np
        best_r = r
        
        for iteration in range(max_iterations):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            if best_r <= 1:
                break
            
            # Try to remove each column by checking if it's a linear combination
            # of others that still satisfies the tensor equation
            removed = False
            for c in range(best_r):
                if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                    break
                
                # Try removing column c
                mask = [i for i in range(best_r) if i != c]
                if len(mask) < best_r - 1:
                    continue
                    
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                # Check if still valid
                if check_valid_fast(U_test, V_test, W_test, tol=1e-8):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_r -= 1
                    removed = True
                    break
            
            if not removed:
                # Try merging two columns with a random linear combination
                c1 = random.randint(0, best_r - 1)
                c2 = random.randint(0, best_r - 1)
                if c1 == c2:
                    continue
                
                # Try several alpha values
                found = False
                for alpha in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
                    U_test = best_U.copy()
                    V_test = best_V.copy()
                    W_test = best_W.copy()
                    
                    U_test[:, c1] += alpha * best_U[:, c2]
                    V_test[:, c1] += alpha * best_V[:, c2]
                    W_test[:, c1] += alpha * best_W[:, c2]
                    
                    mask = [i for i in range(best_r) if i != c2]
                    U_test = U_test[:, mask]
                    V_test = V_test[:, mask]
                    W_test = W_test[:, mask]
                    
                    if check_valid_fast(U_test, V_test, W_test, tol=1e-8):
                        best_U = U_test
                        best_V = V_test
                        best_W = W_test
                        best_r -= 1
                        found = True
                        break
                
                if not found and iteration > 100:
                    # If no progress after many iterations, try a different strategy
                    # Random perturbation and re-check
                    pass
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist()

    def try_joint_decomposition():
        """
        Try to find a joint decomposition that shares rank-1 terms between
        the 4x4x4 and 4x4x3 blocks.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        U4_np = np.array(U4, dtype=float)
        V4_np = np.array(V4, dtype=float)
        W4_np = np.array(W4, dtype=float)
        
        # The 4x4x3 schoolbook part
        U3 = [[0] * 48 for _ in range(16)]
        V3 = [[0] * 48 for _ in range(28)]
        W3 = [[0] * 48 for _ in range(28)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U3[i*4 + j][col] = 1
                    V3[j*7 + k][col] = 1
                    W3[k*4 + i][col] = 1
                    col += 1
        
        U3_np = np.array(U3, dtype=float)
        V3_np = np.array(V3, dtype=float)
        W3_np = np.array(W3, dtype=float)
        
        # Try to find columns in the 4x4x3 part that can be absorbed into the 4x4x4 part
        # by modifying the 4x4x4 decomposition
        best_total = r4 + 48
        best_U = np.hstack([U4_np, U3_np])
        best_V = np.hstack([V4_np, V3_np])
        best_W = np.hstack([W4_np, W3_np])
        
        # Try to eliminate columns from the combined decomposition
        for attempt in range(500):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            r = best_U.shape[1]
            if r <= 1:
                break
            
            # Pick a random column to try to eliminate
            c = random.randint(0, r - 1)
            
            # Try to express column c as a linear combination of others
            # that maintains the tensor equation
            # This is complex, so try simple: set column c to zero and see if
            # we can compensate by modifying other columns
            
            # Alternative: try removing column c and checking validity
            mask = [i for i in range(r) if i != c]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            if check_valid_fast(U_test, V_test, W_test, tol=1e-8):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_total = r - 1
                continue
            
            # Try merging with another random column
            c2 = random.randint(0, r - 1)
            if c2 == c:
                continue
            
            for alpha in [-1.0, -0.5, 0.5, 1.0]:
                U_test = best_U.copy()
                V_test = best_V.copy()
                W_test = best_W.copy()
                
                U_test[:, c] += alpha * best_U[:, c2]
                V_test[:, c] += alpha * best_V[:, c2]
                W_test[:, c] += alpha * best_W[:, c2]
                
                mask = [i for i in range(r) if i != c2]
                U_test = U_test[:, mask]
                V_test = V_test[:, mask]
                W_test = W_test[:, mask]
                
                if check_valid_fast(U_test, V_test, W_test, tol=1e-8):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_total = r - 1
                    break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_total

    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the Kronecker product approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception as e:
            pass
    
    # Try joint decomposition
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_joint, V_joint, W_joint, r_joint = try_joint_decomposition()
            
            if check_valid(U_joint, V_joint, W_joint):
                if r_joint < best_rank:
                    best_result = {"U": U_joint, "V": V_joint, "W": W_joint}
                    best_rank = r_joint
        except Exception as e:
            pass
    
    # Try greedy rank reduction on the best result so far
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_red, V_red, W_red = reduce_rank_greedy(
                best_result["U"], best_result["V"], best_result["W"],
                max_iterations=1000
            )
            r_red = len(U_red[0])
            
            if check_valid(U_red, V_red, W_red):
                if r_red < best_rank:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = r_red
        except Exception as e:
            pass
    
    return best_result
