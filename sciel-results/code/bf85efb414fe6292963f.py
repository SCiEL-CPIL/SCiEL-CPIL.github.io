def add_strassen(U, V, W, I, J, K, term_idx):
    # Adds 7 terms for A_{I,J} B_{J,K} to the global U, V, W.
    # The output C_{I,K} receives contributions from these terms.
    # We need to know the coefficients of M1..M7 in C_{I,K}.
    # But C_{I,K} is the sum of A_{I,0}B_{0,K} and A_{I,1}B_{1,K}.
    # So for a fixed I, K, we call this for J=0 and J=1.
    # The coefficients for M1..M7 are the same for both J's?
    # Yes, the Strassen formula is the same.
    # So we can just add the 7 terms with the standard Strassen coefficients.
    
    # Coefficients for M1..M7 in C:
    # C00: 1, 0, 0, 1, -1, 0, 1
    # C01: 0, 0, 1, 0, 1, 0, 0
    # C10: 0, 1, 0, 1, 0, 0, 0
    # C11: 1, -1, 1, 0, 0, 1, 1
    
    # We add to W with these coefficients.
    
    def u_idx(r, c):
        i = 2*I + r
        j = 2*J + c
        return i*4 + j
        
    def v_idx(r, c):
        j = 2*J + r
        k = 2*K + c
        return j*8 + k
        
    def w_idx(r, c):
        k = 2*K + c
        i = 2*I + r
        return k*4 + i
        
    # M1
    U[u_idx(0,0), term_idx] += 1
    U[u_idx(1,1), term_idx] += 1
    V[v_idx(0,0), term_idx] += 1
    V[v_idx(1,1), term_idx] += 1
    W[w_idx(0,0), term_idx] += 1
    W[w_idx(1,1), term_idx] += 1
    term_idx += 1
    
    # M2
    U[u_idx(1,0), term_idx] += 1
    U[u_idx(1,1), term_idx] += 1
    V[v_idx(0,0), term_idx] += 1
    W[w_idx(1,0), term_idx] += 1
    W[w_idx(1,1), term_idx] += -1
    term_idx += 1
    
    # M3
    U[u_idx(0,0), term_idx] += 1
    V[v_idx(0,1), term_idx] += 1
    V[v_idx(1,1), term_idx] += -1
    W[w_idx(0,1), term_idx] += 1
    W[w_idx(1,1), term_idx] += 1
    term_idx += 1
    
    # M4
    U[u_idx(1,1), term_idx] += 1
    V[v_idx(1,0), term_idx] += 1
    V[v_idx(0,0), term_idx] += -1
    W[w_idx(0,0), term_idx] += 1
    W[w_idx(1,0), term_idx] += 1
    term_idx += 1
    
    # M5
    U[u_idx(0,0), term_idx] += 1
    U[u_idx(0,1), term_idx] += 1
    V[v_idx(1,1), term_idx] += 1
    W[w_idx(0,1), term_idx] += -1
    W[w_idx(1,1), term_idx] += 1
    term_idx += 1
    
    # M6
    U[u_idx(1,0), term_idx] += 1
    U[u_idx(0,0), term_idx] += -1
    V[v_idx(0,0), term_idx] += 1
    V[v_idx(0,1), term_idx] += 1
    W[w_idx(1,1), term_idx] += 1
    term_idx += 1
    
    # M7
    U[u_idx(0,1), term_idx] += 1
    U[u_idx(1,1), term_idx] += -1
    V[v_idx(1,0), term_idx] += 1
    V[v_idx(1,1), term_idx] += 1
    W[w_idx(0,0), term_idx] += 1
    W[w_idx(1,0), term_idx] += -1
    W[w_idx(1,1), term_idx] += 1
    term_idx += 1
    
    return term_idx
