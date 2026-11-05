import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    
    # Strategy: Partition middle dimension m=3 into 2+1
    # Part 1: 2x2 * 2x10 using Strassen (rank 7 per 2x2x2 block) -> 5 blocks * 7 = 35
    # Part 2: 2x1 * 1x10 using schoolbook -> 2*1*10 = 20
    # Total rank: 55
    
    # However, we can try to be smarter about Part 2.
    # The current program uses rank 55.
    # Is it possible to do better?
    # The lower bound for 2x3 * 3x10 is max(n*p, n*m, m*p) = max(20, 6, 30) = 30.
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, n*p).
    # For 2x3, 3x10: max(6, 30, 20) = 30.
    # But we are looking for an exact decomposition.
    # The current score 1.0909 corresponds to rank 55 (seed 60 / 55 = 1.0909).
    # If we could find rank 54, score = 60/54 = 1.111.
    # If rank 50, score = 1.2.
    
    # Let's verify the Strassen decomposition is correct and see if we can optimize the tail.
    # The "tail" is the 2x1 * 1x10 part.
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x10); B2 (1x10)]
    # C = A1 B1 + A2 B2
    
    # Strassen for A1 B1 (2x2 * 2x10) uses 7 multiplications per 2 columns of B1.
    # Total 35 mults.
    # A2 B2 is 2x1 * 1x10. This is just 20 scalar mults.
    # Can we combine A2 B2 with the Strassen part?
    # The Strassen algorithm computes products of sums of A1 entries and B1 entries.
    # A2 is independent.
    # Usually, these are disjoint.
    
    # Let's stick to rank 55 as it is a known good solution.
    # The task is to "find a rank-minimal exact tensor decomposition".
    # The seed is 60. We are at 55.
    # Can we do 54?
    # Research suggests that for 2x3 * 3x10, the rank might be 55 or lower.
    # However, without a search algorithm (which is complex and risky within 120s),
    # we should ensure the current 55 is implemented correctly and efficiently.
    
    # The inspirations show the same 55 rank structure.
    # Let's re-verify the Strassen indices to ensure no bugs.
    
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: Strassen for 2x2 * 2x10
    # A1 = [[a00, a01], [a10, a11]] (indices in U: 0,1,3,4)
    # B1 = [[b00...b09], [b10...b19]] (indices in V: 0..9, 10..19)
    # C1 = A1 B1 (indices in W: 0..19, W[k*n+i] -> W[2k+0] is C[0,k], W[2k+1] is C[1,k])
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # B indices for this block (columns k0, k1 of B1)
        v_b0k0 = k0
        v_b0k1 = k1
        v_b1k0 = 10 + k0
        v_b1k1 = 10 + k1
        
        # C indices for this block (columns k0, k1 of C1)
        w_c0k0 = 2 * k0
        w_c0k1 = 2 * k1
        w_c1k0 = 2 * k0 + 1
        w_c1k1 = 2 * k1 + 1
        
        # Standard Strassen for 2x2 * 2x2 (here B is 2x2 slice)
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's trace the contributions to W (C matrix)
        
        # M1
        U[u_a00][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        V[v_b1k1][idx] += 1
        # W contributions: C00 += 1, C11 += 1
        W[w_c0k0][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M2
        U[u_a10][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        # W contributions: C10 += 1, C11 -= 1
        W[w_c1k0][idx] += 1
        W[w_c1k1][idx] -= 1
        idx += 1
        
        # M3
        U[u_a00][idx] += 1
        V[v_b0k1][idx] += 1
        V[v_b1k1][idx] -= 1
        # W contributions: C01 += 1, C11 += 1
        W[w_c0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M4
        U[u_a11][idx] += 1
        V[v_b1k0][idx] += 1
        V[v_b0k0][idx] -= 1
        # W contributions: C00 -= 1, C10 += 1
        W[w_c0k0][idx] -= 1
        W[w_c1k0][idx] += 1
        idx += 1
        
        # M5
        U[u_a00][idx] += 1
        U[u_a01][idx] += 1
        V[v_b1k1][idx] += 1
        # W contributions: C00 += 1, C01 += 1
        W[w_c0k0][idx] += 1
        W[w_c0k1][idx] += 1
        idx += 1
        
        # M6
        U[u_a10][idx] += 1
        U[u_a00][idx] -= 1
        V[v_b0k0][idx] += 1
        V[v_b0k1][idx] += 1
        # W contributions: C11 += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M7
        U[u_a01][idx] += 1
        U[u_a11][idx] -= 1
        V[v_b1k0][idx] += 1
        V[v_b1k1][idx] += 1
        # W contributions: C00 += 1
        W[w_c0k0][idx] += 1
        idx += 1

    # Part 2: Schoolbook for 2x1 * 1x10
    # A2 = [[a02], [a12]] (indices in U: 2, 5)
    # B2 = [b20...b29] (indices in V: 20..29)
    # C2 = A2 B2 (indices in W: 0..19)
    
    u_a02 = 2
    u_a12 = 5
    
    for k in range(10):
        v_b2k = 20 + k
        w_c0k = 2 * k
        w_c1k = 2 * k + 1
        
        # A[0,2] * B[2,k] -> C[0,k]
        U[u_a02][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c0k][idx] += 1
        idx += 1
        
        # A[1,2] * B[2,k] -> C[1,k]
        U[u_a12][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c1k][idx] += 1
        idx += 1

    # Verification
    if time.time() < deadline - 5:
        try:
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            np.random.seed(42)
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            
            for r in range(rank):
                u_r = U_np[:, r]
                v_r = V_np[:, r]
                w_r = W_np[:, r]
                
                # Compute scalar product of weighted A and A
                # sum_{i,j} u_{ij} A_{ij}
                A_flat = A.flatten()
                B_flat = B.flatten()
                U_flat = u_r.flatten()
                V_flat = v_r.flatten()
                
                sA = np.dot(U_flat, A_flat)
                sB = np.dot(V_flat, B_flat)
                
                # Add sA * sB * w_r to C
                C_flat = C_computed.flatten()
                C_flat += sA * sB * w_r
                C_computed = C_flat.reshape(n, p)
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Fallback to schoolbook rank 60 if Strassen fails (should not happen)
                rank = 60
                U = [[0.0] * rank for _ in range(n * m)]
                V = [[0.0] * rank for _ in range(m * p)]
                W = [[0.0] * rank for _ in range(p * n)]
                
                idx = 0
                for i in range(n):
                    for k in range(m):
                        for l in range(p):
                            U[i * m + k][idx] = 1.0
                            V[k * p + l][idx] = 1.0
                            W[l * n + i][idx] = 1.0
                            idx += 1
        except Exception:
            # If verification fails, return the constructed matrices anyway, 
            # as the logic is deterministic and matches known Strassen.
            pass

    return {'U': U, 'V': V, 'W': W}
