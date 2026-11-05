import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: Strassen for 2x2 * 2x10 (5 blocks of 2x2 * 2x2)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        v_b0k0 = k0
        v_b0k1 = k1
        v_b1k0 = 10 + k0
        v_b1k1 = 10 + k1
        
        w_c0k0 = 2 * k0
        w_c0k1 = 2 * k1
        w_c1k0 = 2 * k0 + 1
        w_c1k1 = 2 * k1 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U[u_a00][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U[u_a10][idx] += 1
        U[u_a11][idx] += 1
        V[v_b0k0][idx] += 1
        W[w_c1k0][idx] += 1
        W[w_c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U[u_a00][idx] += 1
        V[v_b0k1][idx] += 1
        V[v_b1k1][idx] -= 1
        W[w_c0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U[u_a11][idx] += 1
        V[v_b1k0][idx] += 1
        V[v_b0k0][idx] -= 1
        W[w_c0k0][idx] += 1
        W[w_c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U[u_a00][idx] += 1
        U[u_a01][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] -= 1
        W[w_c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U[u_a10][idx] += 1
        U[u_a00][idx] -= 1
        V[v_b0k0][idx] += 1
        V[v_b0k1][idx] += 1
        W[w_c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U[u_a01][idx] += 1
        U[u_a11][idx] -= 1
        V[v_b1k0][idx] += 1
        V[v_b1k1][idx] += 1
        W[w_c0k0][idx] += 1
        idx += 1

    # Part 2: Schoolbook for 2x1 * 1x10
    u_a02 = 2
    u_a12 = 5
    
    for k in range(10):
        v_b2k = 20 + k
        w_c0k = 2 * k
        w_c1k = 2 * k + 1
        
        # A[0,2] * B[2,k]
        U[u_a02][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c0k][idx] += 1
        idx += 1
        
        # A[1,2] * B[2,k]
        U[u_a12][idx] += 1
        V[v_b2k][idx] += 1
        W[w_c1k][idx] += 1
        idx += 1

    return {'U': U, 'V': V, 'W': W}
