import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    def build_strassen_solution():
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
            U[0][r] += 1; U[3][r] += 1
            V[v_e][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1; W[w_c11][r] += 1
            r += 1
            # m2 = (c+d)e
            U[2][r] += 1; U[3][r] += 1
            V[v_e][r] += 1
            W[w_c10][r] += 1; W[w_c11][r] += -1
            r += 1
            # m3 = a(f-h)
            U[0][r] += 1
            V[v_f][r] += 1; V[v_h][r] += -1
            W[w_c01][r] += 1; W[w_c11][r] += 1
            r += 1
            # m4 = d(g-e)
            U[3][r] += 1
            V[v_g][r] += 1; V[v_e][r] += -1
            W[w_c00][r] += 1; W[w_c10][r] += 1
            r += 1
            # m5 = (a+b)h
            U[0][r] += 1; U[1][r] += 1
            V[v_h][r] += 1
            W[w_c00][r] += -1; W[w_c01][r] += 1
            r += 1
            # m6 = (c-a)(e+f)
            U[2][r] += 1; U[0][r] += -1
            V[v_e][r] += 1; V[v_f][r] += 1
            W[w_c11][r] += 1
            r += 1
            # m7 = (b-d)(g+h)
            U[1][r] += 1; U[3][r] += -1
            V[v_g][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1
            r += 1
        
        # Last column k=10: schoolbook
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
        U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
        U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
        U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        n, m, p = 2, 2, 11
        rank = len(U[0])
        # Check dimensions
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        for i in range(len(U)):
            if len(U[i]) != rank: return False
        for i in range(len(V)):
            if len(V[i]) != rank: return False
        for i in range(len(W)):
            if len(W[i]) != rank: return False
        
        # Test with random integer inputs
        import random
        random.seed(42)
        for _ in range(5):
            A = [[random.randint(-5,5) for _ in range(2)] for _ in range(2)]
            B = [[random.randint(-5,5) for _ in range(11)] for _ in range(2)]
            
            # Compute C = A * B directly
            C_direct = [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(11)] for i in range(2)]
            
            # Compute C via decomposition
            C_decomp = [[0]*11 for _ in range(2)]
            for i in range(2):
                for k in range(11):
                    total = 0
                    for t in range(rank):
                        u_val = 0
                        for j in range(2):
                            u_val += U[i*2 + j][t] * A[i][j]
                        v_val = 0
                        for j in range(2):
                            v_val += V[j*11 + k][t] * B[j][k]
                        w_val = W[k*2 + i][t]
                        total += w_val * u_val * v_val
                    C_decomp[i][k] = total
                    
                    if C_decomp[i][k] != C_direct[i][k]:
                        return False
        return True
    
    # Build and verify the rank-39 solution
    best = build_strassen_solution()
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback: schoolbook rank-44
        rank = 44
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(22)]
        W = [[0]*rank for _ in range(22)]
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(11):
                    U[i*2+j][r] = 1
                    V[j*11+k][r] = 1
                    W[k*2+i][r] = 1
                    r += 1
        best = {"U": U, "V": V, "W": W}
    
    return best
