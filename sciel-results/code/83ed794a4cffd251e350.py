import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Laderman block for columns k, k+1
    def add_laderman(k, k1):
        nonlocal col
        # 1. (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[0*7+k][col] = 1; V[1*7+k1][col] = 1
        # C0k = +1, C1k+1 = +1? 
        # C0k = m1 + m2 - m3 + m5
        # C1k+1 = m1 - m4 + m2
        # Wait, let's just map the multipliers to W later?
        # No, W is the output factor.
        # The decomposition is C = sum_r U_r V_r W_r^T ?
        # Actually, C[c] = sum_r U[a] V[b] W[c] is not the form.
        # The form is: C = sum_r (U[:,r] * V[:,r]) * W[:,r] ? No.
        # The standard tensor decomposition:
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # So for each multiplication r, we define vectors u_r, v_r, w_r.
        # u_r has non-zeros at A indices.
        # v_r has non-zeros at B indices.
        # w_r has non-zeros at C indices.
        
        # For Laderman:
        # m1 = (a+d)(e+h)
        # Contribution to C:
        # C0k += 1 * m1
        # C1k+1 += 1 * m1
        # So w1 has 1 at C0k and 1 at C1k+1.
        
        # m2 = (a-d)(e-h)
        # C0k += 1 * m2
        # C0k+1 += -1 * m2
        # C1k+1 += 1 * m2
        # w2 has 1 at C0k, -1 at C0k+1, 1 at C1k+1.
        
        # m3 = (a+b)(e+g)
        # C0k += -1 * m3
        # C0k+1 += 1 * m3
        # w3 has -1 at C0k, 1 at C0k+1.
        
        # m4 = (c+d)(e+h)
        # C1k += 1 * m4
        # C1k+1 += -1 * m4
        # w4 has 1 at C1k, -1 at C1k+1.
        
        # m5 = (a+c)(e+g)
        # C0k += 1 * m5
        # C1k += -1 * m5
        # w5 has 1 at C0k, -1 at C1k.
        
        # m6 = (b-d)(f-h)
        # C1k += -1 * m6
        # C0k+1 += 1 * m6
        # w6 has -1 at C1k, 1 at C0k+1.
        
        # m7 = (c-a)(g-h)
        # C1k += 1 * m7
        # C0k+1 += -1 * m7
        # w7 has 1 at C1k, -1 at C0k+1.
        
        # Indices:
        # A: a=0, b=1, c=2, d=3
        # B: e=B0k, f=B0k1, g=B1k, h=B1k1
        #    B indices: 0*7+k, 0*7+k1, 1*7+k, 1*7+k1
        # C: C0k = k*2+0, C1k = k*2+1, C0k1 = k1*2+0, C1k1 = k1*2+1
        
        # m1
        U[0][col] = 1; U[3][col] = 1
        V[0*7+k][col] = 1; V[1*7+k1][col] = 1
        W[k*2+0][col] = 1; W[k1*2+1][col] = 1
        col += 1
        
        # m2
        U[0][col] = 1; U[3][col] = -1
        V[0*7+k][col] = 1; V[1*7+k1][col] = -1
        W[k*2+0][col] = 1; W[k1*2+0][col] = -1; W[k1*2+1][col] = 1
        col += 1
        
        # m3
        U[0][col] = 1; U[1][col] = 1
        V[0*7+k][col] = 1; V[1*7+k][col] = 1
        W[k*2+0][col] = -1; W[k1*2+0][col] = 1
        col += 1
        
        # m4
        U[2][col] = 1; U[3][col] = 1
        V[0*7+k][col] = 1; V[1*7+k1][col] = 1
        W[k*2+1][col] = 1; W[k1*2+1][col] = -1
        col += 1
        
        # m5
        U[0][col] = 1; U[2][col] = 1
        V[0*7+k][col] = 1; V[1*7+k][col] = 1
        W[k*2+0][col] = 1; W[k*2+1][col] = -1
        col += 1
        
        # m6
        U[1][col] = 1; U[3][col] = -1
        V[0*7+k1][col] = 1; V[1*7+k1][col] = -1
        W[k*2+1][col] = -1; W[k1*2+0][col] = 1
        col += 1
        
        # m7
        U[2][col] = 1; U[0][col] = -1
        V[1*7+k][col] = 1; V[1*7+k1][col] = -1
        W[k*2+1][col] = 1; W[k1*2+0][col] = -1
        col += 1

    # Helper to add a Rank-4 block for single column k
    def add_single(k):
        nonlocal col
        # m1 = (a+b)(e+g)
        # C0k += 0.5 m1
        # m2 = (a-b)(e-g)
        # C0k += 0.5 m2
        # m3 = (c+d)(e+g)
        # C1k += 0.5 m3
        # m4 = (c-d)(e-g)
        # C1k += 0.5 m4
        
        # m1
        U[0][col] = 1; U[1][col] = 1
        V[0*7+k][col] = 1; V[1*7+k][col] = 1
        W[k*2+0][col] = 0.5
        col += 1
        
        # m2
        U[0][col] = 1; U[1][col] = -1
        V[0*7+k][col] = 1; V[1*7+k][col] = -1
        W[k*2+0][col] = 0.5
        col += 1
        
        # m3
        U[2][col] = 1; U[3][col] = 1
        V[0*7+k][col] = 1; V[1*7+k][col] = 1
        W[k*2+1][col] = 0.5
        col += 1
        
        # m4
        U[2][col] = 1; U[3][col] = -1
        V[0*7+k][col] = 1; V[1*7+k][col] = -1
        W[k*2+1][col] = 0.5
        col += 1

    add_laderman(0, 1)
    add_laderman(2, 3)
    add_laderman(4, 5)
    add_single(6)
    
    return {"U": U, "V": V, "W": W}
