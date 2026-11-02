"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_tensor(u, v, w):
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)

    # Strassen decomposition for 2x2 x 2x2 blocks
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices for pair j, j+1:
    # B[0,j] -> j, B[0,j+1] -> j+1
    # B[1,j] -> p+j, B[1,j+1] -> p+j+1
    # C indices:
    # C[j,0] -> 2j, C[j,1] -> 2j+1
    # C[j+1,0] -> 2j+2, C[j+1,1] -> 2j+3

    for pair_idx in range(5):
        j = 2 * pair_idx
        
        # M1 = (A00+A11)(B[0,j]+B[1,j])
        u = [1, 0, 0, 1]
        v = [0]*p; v[j]=1; v[p+j]=1
        w = [0]*(p*n); w[2*j]=1; w[2*j+1]=1; w[2*j+2]=1; w[2*j+3]=1
        add_tensor(u, v, w)
        
        # M2 = (A10+A11)B[0,j]
        u = [0, 0, 1, 1]
        v = [0]*p; v[j]=1
        w = [0]*(p*n); w[2*j+1]=-1; w[2*j+2]=1
        add_tensor(u, v, w)
        
        # M3 = A00(B[1,j]-B[0,j])
        u = [1, 0, 0, 0]
        v = [0]*p; v[j]=-1; v[p+j]=1
        w = [0]*(p*n); w[2*j+1]=1; w[2*j+3]=-1
        add_tensor(u, v, w)
        
        # M4 = A11(B[1,j+1]-B[0,j+1])
        u = [0, 0, 0, 1]
        v = [0]*p; v[j+1]=-1; v[p+j+1]=1
        w = [0]*(p*n); w[2*j]=1; w[2*j+2]=-1
        add_tensor(u, v, w)
        
        # M5 = (A00+A01)B[1,j+1]
        u = [1, 1, 0, 0]
        v = [0]*p; v[p+j+1]=1
        w = [0]*(p*n); w[2*j]=-1; w[2*j+1]=1
        add_tensor(u, v, w)
        
        # M6 = (A10-A00)(B[0,j]+B[0,j+1])
        u = [0, 0, 1, -1]
        v = [0]*p; v[j]=1; v[j+1]=1
        w = [0]*(p*n); w[2*j+2]=1
        add_tensor(u, v, w)
        
        # M7 = (A01-A11)(B[0,j+1]+B[1,j+1])
        u = [0, 1, 0, -1]
        v = [0]*p; v[j+1]=1; v[p+j+1]=1
        w = [0]*(p*n); w[2*j]=1
        add_tensor(u, v, w)
        
    # Last column j=10 (Schoolbook)
    j = 10
    u = [1, 0, 0, 0]; v = [0]*p; v[j]=1; w = [0]*(p*n); w[2*j]=1
    add_tensor(u, v, w)
    u = [0, 1, 0, 0]; v = [0]*p; v[p+j]=1; w = [0]*(p*n); w[2*j]=1
    add_tensor(u, v, w)
    u = [0, 0, 1, 0]; v = [0]*p; v[j]=1; w = [0]*(p*n); w[2*j+1]=1
    add_tensor(u, v, w)
    u = [0, 0, 0, 1]; v = [0]*p; v[p+j]=1; w = [0]*(p*n); w[2*j+1]=1
    add_tensor(u, v, w)

    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
