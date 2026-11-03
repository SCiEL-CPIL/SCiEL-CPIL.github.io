import numpy as np

def construct():
    n, m, p = 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    R = 54
    U = np.zeros((nm, R), dtype=np.float64)
    V = np.zeros((mp, R), dtype=np.float64)
    W = np.zeros((pn, R), dtype=np.float64)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices for 2x2 A
        # b_idx: list of 4 indices for 2x2 B
        # c_idx: list of 4 indices for 2x2 C
        # a = [[a0, a1], [a2, a3]]
        # b = [[b0, b1], [b2, b3]]
        # c = [[c0, c1], [c2, c3]]
        
        # M1 = (a0+a3)(b0+b3)
        add_term(a_idx[0], b_idx[0], c_idx[0])
        add_term(a_idx[0], b_idx[3], c_idx[0])
        add_term(a_idx[3], b_idx[0], c_idx[0])
        add_term(a_idx[3], b_idx[3], c_idx[0])
        
        # M2 = (a2+a3)b0
        add_term(a_idx[2], b_idx[0], c_idx[2])
        add_term(a_idx[3], b_idx[0], c_idx[2])
        
        # M3 = a0(b1-b3)
        add_term(a_idx[0], b_idx[1], c_idx[1])
        add_term(a_idx[0], b_idx[3], c_idx[1], -1)
        
        # M4 = a3(b2-b0)
        add_term(a_idx[3], b_idx[2], c_idx[2])
        add_term(a_idx[3], b_idx[0], c_idx[2], -1)
        
        # M5 = (a0+a1)b3
        add_term(a_idx[0], b_idx[3], c_idx[0])
        add_term(a_idx[1], b_idx[3], c_idx[0])
        
        # M6 = (a2-a0)(b0+b1)
        add_term(a_idx[2], b_idx[0], c_idx[2])
        add_term(a_idx[2], b_idx[1], c_idx[2])
        add_term(a_idx[0], b_idx[0], c_idx[2], -1)
        add_term(a_idx[0], b_idx[1], c_idx[2], -1)
        
        # M7 = (a1-a3)(b2+b3)
        add_term(a_idx[1], b_idx[2], c_idx[1])
        add_term(a_idx[1], b_idx[3], c_idx[1])
        add_term(a_idx[3], b_idx[2], c_idx[1], -1)
        add_term(a_idx[3], b_idx[3], c_idx[1], -1)
        
        # Wait, the output combination is not just adding to W.
        # The W matrix accumulates the coefficients for C.
        # My add_term adds to U, V, W simultaneously.
        # This is correct for the tensor decomposition.
        # T = sum u v w.
        # If I split M1 into 4 terms, I am adding 4 rank-1 tensors.
        # This increases the rank!
        # Strassen uses 7 multiplications, but the inputs are linear combinations.
        # The rank is 7.
        # So I should not expand the linear combinations into separate rank-1 terms in the tensor U,V,W.
        # Instead, each "multiplication" M_k corresponds to one column in U,V,W.
        # U column k has the coefficients for A in M_k.
        # V column k has the coefficients for B in M_k.
        # W column k has the coefficients for C in M_k.
        
        # So for M1 = (a0+a3)(b0+b3):
        # U col: [1, 0, 1, 0] at a_idx positions.
        # V col: [1, 0, 1, 0] at b_idx positions.
        # W col: [1, 0, 0, 0] at c_idx positions? No.
        # M1 contributes to c0, c1, c2, c3.
        # c0 += M1
        # c1 += M1
        # c2 += M1
        # c3 += M1
        # So W col has 1 at all c_idx?
        # No, the contribution is M1 * (1, 1, 1, 1) for C?
        # Z00 = M1 + M4 - M5 + M7
        # So M1 contributes 1 to Z00.
        # Z01 = M3 + M5
        # M1 contributes 0 to Z01.
        
        # So for M1:
        # U: a0=1, a3=1.
        # V: b0=1, b3=1.
        # W: c0=1, c1=0, c2=0, c3=0.
        
        # This is 1 rank-1 term.
        
        pass
