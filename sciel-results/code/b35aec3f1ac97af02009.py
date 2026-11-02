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
    
    def try_random_elimination(U, V, W, max_iter=200, seed=42):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        If two terms t1 and t2 satisfy that their contribution can be combined,
        we can potentially eliminate one.
        """
        import copy
        U = copy.deepcopy(U)
        V = copy.deepcopy(V)
        W = copy.deepcopy(W)
        r = len(U[0])
        rng = random.Random(seed)
        
        for iteration in range(max_iter):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            if r <= 1:
                break
            
            # Pick two random distinct columns
            idx = rng.sample(range(r), 2)
            a, b = idx
            
            # Try to find alpha such that term a + alpha * term b = 0 contribution
            # i.e., U[i,a]*V[j,a]*W[k,a] + alpha * U[i,b]*V[j,b]*W[k,b] = 0 for all i,j,k
            # This means U[i,a]*V[j,a]*W[k,a] = -alpha * U[i,b]*V[j,b]*W[k,b]
            # So the tensors U_a ⊗ V_a ⊗ W_a and U_b ⊗ V_b ⊗ W_b must be proportional
            
            # Check if they are proportional by finding alpha from one non-zero entry
            alpha = None
            for i in range(16):
                for j in range(28):
                    for k in range(28):
                        # Need to map to actual indices
                        pass
            
            # Simpler: check if the outer products are proportional
            # Build the tensor for each term
            ta = np.zeros((16, 28, 28))
            tb = np.zeros((16, 28, 28))
            for i in range(16):
                for j in range(28):
                    for k in range(28):
                        ta[i,j,k] = U[i][a] * V[j][a] * W[k][a]
                        tb[i,j,k] = U[i][b] * V[j][b] * W[k][b]
            
            # Check proportionality
            nonzero_a = np.nonzero(ta)
            nonzero_b = np.nonzero(tb)
            
            if len(nonzero_a[0]) == 0 or len(nonzero_b[0]) == 0:
                continue
            
            # Find alpha
            found_alpha = None
            for idx2 in range(len(nonzero_a[0])):
                i, j, k = nonzero_a[0][idx2], nonzero_a[1][idx2], nonzero_a[2][idx2]
                if abs(tb[i,j,k]) > 1e-12:
                    found_alpha = -ta[i,j,k] / tb[i,j,k]
                    break
            
            if found_alpha is None:
                continue
            
            # Verify proportionality
            valid = True
            for i in range(16):
                for j in range(28):
                    for k in range(28):
                        if abs(ta[i,j,k] + found_alpha * tb[i,j,k]) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Term a = -alpha * term b, so we can eliminate term a
                # Remove column a
                U = [row[:a] + row[a+1:] for row in U]
                V = [row[:a] + row[a+1:] for row in V]
                W = [row[:a] + row[a+1:] for row in W]
                r -= 1
                # Adjust index b if needed
                if b > a:
                    b -= 1
        
        return U, V, W, r
    
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
                    
                    # Try random elimination to reduce rank further
                    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
                        U_red, V_red, W_red, r_red = try_random_elimination(
                            U_kron, V_kron, W_kron, max_iter=500, seed=42)
                        if r_red < best_rank:
                            if check_valid(U_red, V_red, W_red):
                                best_result = {"U": U_red, "V": V_red, "W": W_red}
                                best_rank = r_red
        except Exception as e:
            pass
    
    return best_result
