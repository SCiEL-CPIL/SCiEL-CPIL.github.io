import numpy as np

def construct():
    # ... existing code ...
    
    # Remove zero columns
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    keep = []
    for i in range(rank):
        if not (np.all(U_arr[:, i] == 0) and np.all(V_arr[:, i] == 0) and np.all(W_arr[:, i] == 0)):
            keep.append(i)
            
    if len(keep) < rank:
        U = [U[i] for i in keep] # Wait, U is row-major. U[r, c].
        # U is list of lists. U[i][j].
        # We need to keep column j.
        new_U = [[U[i][j] for j in keep] for i in range(dim_u)]
        new_V = [[V[i][j] for j in keep] for i in range(dim_v)]
        new_W = [[W[i][j] for j in keep] for i in range(dim_w)]
        return {"U": new_U, "V": new_V, "W": new_W}
        
    return {"U": U, "V": V, "W": W}
