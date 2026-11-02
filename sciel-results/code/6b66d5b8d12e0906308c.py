import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a multiplication term
    # term: (u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs)
    # But it's easier to just loop through the 11 terms per group
    
    for g in range(5):
        base_t = g * 11
        
        # Global indices for A (U)
        # A is 2x3. Index i*m + j.
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1], x=A[0,2], y=A[1,2]
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        u_x = 0 * m + 2
        u_y = 1 * m + 2
        
        # Global indices for B (V)
        # B is 3x10. Index j*p + k.
        # Group g uses cols 2g, 2g+1.
        # e=B[0,2g], f=B[0,2g+1], g_b=B[1,2g], h=B[1,2g+1], u=B[2,2g], v=B[2,2g+1]
        v_e = 0 * p + 2*g
        v_f = 0 * p + 2*g + 1
        v_g = 1 * p + 2*g
        v_h = 1 * p + 2*g + 1
        v_u = 2 * p + 2*g
        v_v = 2 * p + 2*g + 1
        
        # Global indices for C (W)
        # C is 2x10. Index k*n + i.
        # C[0,2g], C[0,2g+1], C[1,2g], C[1,2g+1]
        w_c11 = (2*g) * n + 0
        w_c12 = (2*g + 1) * n + 0
        w_c21 = (2*g) * n + 1
        w_c22 = (2*g + 1) * n + 1
        
        # Define the 11 multiplications
        # Format: (t_offset, u_idx, u_val, v_idx, v_val, w_idx, w_val)
        # We need to add to U, V, W at column base_t + t_offset
        
        terms = []
        
        # Strassen part (7 terms)
        # m1 = (a+d)(e+h)
        terms.append((0, u_a, 1, v_e, 1, w_c11, 1))
        terms.append((0, u_d, 1, v_h, 1, w_c22, 1))
        # Note: m1 contributes to C11 and C22.
        # We need to add both W contributions.
        # My term structure above allows only one W contribution per tuple if I'm not careful.
        # Let's use a list of (u_idx, u_val, v_idx, v_val, w_idx, w_val) for each t.
        
        # Let's restructure: for each t in 0..10, define U[t], V[t], W[t] entries.
        
        # t=0: m1
        U[u_a][base_t+0] += 1
        U[u_d][base_t+0] += 1
        V[v_e][base_t+0] += 1
        V[v_h][base_t+0] += 1
        W[w_c11][base_t+0] += 1
        W[w_c22][base_t+0] += 1
        
        # t=1: m2 = (c+d)e
        U[u_c][base_t+1] += 1
        U[u_d][base_t+1] += 1
        V[v_e][base_t+1] += 1
        W[w_c21][base_t+1] += 1
        W[w_c22][base_t+1] += -1
        
        # t=2: m3 = a(f-h)
        U[u_a][base_t+2] += 1
        V[v_f][base_t+2] += 1
        V[v_h][base_t+2] += -1
        W[w_c12][base_t+2] += 1
        W[w_c22][base_t+2] += 1
        
        # t=3: m4 = d(g-e)
        U[u_d][base_t+3] += 1
        V[v_g][base_t+3] += 1
        V[v_e][base_t+3] += -1
        W[w_c11][base_t+3] += 1
        W[w_c21][base_t+3] += 1
        
        # t=4: m5 = (a+b)h
        U[u_a][base_t+4] += 1
        U[u_b][base_t+4] += 1
        V[v_h][base_t+4] += 1
        W[w_c11][base_t+4] += -1
        W[w_c12][base_t+4] += 1
        
        # t=5: m6 = (c-a)(e+f)
        U[u_c][base_t+5] += 1
        U[u_a][base_t+5] += -1
        V[v_e][base_t+5] += 1
        V[v_f][base_t+5] += 1
        W[w_c22][base_t+5] += 1
        
        # t=6: m7 = (b-d)(g+h)
        U[u_b][base_t+6] += 1
        U[u_d][base_t+6] += -1
        V[v_g][base_t+6] += 1
        V[v_h][base_t+6] += 1
        W[w_c11][base_t+6] += 1
        
        # Standard part (4 terms)
        # t=7: p1 = x*u
        U[u_x][base_t+7] += 1
        V[v_u][base_t+7] += 1
        W[w_c11][base_t+7] += 1
        
        # t=8: p2 = x*v
        U[u_x][base_t+8] += 1
        V[v_v][base_t+8] += 1
        W[w_c12][base_t+8] += 1
        
        # t=9: p3 = y*u
        U[u_y][base_t+9] += 1
        V[v_u][base_t+9] += 1
        W[w_c21][base_t+9] += 1
        
        # t=10: p4 = y*v
        U[u_y][base_t+10] += 1
        V[v_v][base_t+10] += 1
        W[w_c22][base_t+10] += 1

    return {"U": U, "V": V, "W": W}
