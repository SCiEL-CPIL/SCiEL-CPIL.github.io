import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # T = sum_{c=1}^7 u_c v_c w_c
    # u, v, w are vectors of length 2
    u_strassen = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 1],
        [1, -1],
        [1, 0]
    ]
    v_strassen = [
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    w_strassen = [
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ]
    
    # Build M_4,4,4 decomposition via Kronecker product
    U_444 = []
    V_444 = []
    W_444 = []
    
    for i in range(7):
        for j in range(7):
            u = np.kron(u_strassen[i], u_strassen[j])
            v = np.kron(v_strassen[i], v_strassen[j])
            w = np.kron(w_strassen[i], w_strassen[j])
            U_444.append(u)
            V_444.append(v)
            W_444.append(w)
            
    # Now we have rank 49 for M_4,4,4
    # We need M_4,5,5
    # The tensor T_4,5,5 has entries T[i,j,k] = 1 if j==k? No.
    # T[i,j,k] = 1 if the index j is the contraction index.
    # The decomposition covers j=1..4.
    # We need to add terms for j=5.
    # The part of the tensor for j=5 is T[i,5,k] = 1 for all i,k.
    # This is a sum of e_i \otimes e_5 \otimes e_k.
    # There are 4*5 = 20 such terms.
    
    U_455 = [np.array(u) for u in U_444]
    V_455 = [np.array(v) for v in V_444]
    W_455 = [np.array(w) for w in W_444]
    
    # Extend V to length 5 (pad with 0)
    for idx in range(len(V_455)):
        V_455[idx] = np.append(V_455[idx], 0)
        
    # Add the 20 terms for j=5
    for i in range(4):
        for k in range(5):
            u = np.zeros(4)
            u[i] = 1
            v = np.zeros(5)
            v[4] = 1 # index 5 is 4 in 0-based
            w = np.zeros(5)
            w[k] = 1
            U_455.append(u)
            V_455.append(v)
            W_455.append(w)
            
    rank = len(U_455)
    
    # Convert to lists
    U = [u.tolist() for u in U_455]
    V = [v.tolist() for v in V_455]
    W = [w.tolist() for w in W_455]
    
    return {"U": U, "V": V, "W": W}
