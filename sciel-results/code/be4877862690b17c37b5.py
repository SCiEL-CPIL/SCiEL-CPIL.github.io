import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    # Shapes: U (n*m, rank) = (4, 14), V (m*p, rank) = (8, 14), W (p*n, rank) = (8, 14)
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
            
    # Indices:
    # U (A): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V (B): 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W (C): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    
    # The hardcoded solution from the current program
    # Left block (B[:, 0:2]) corresponds to k=0,1
    # Right block (B[:, 2:4]) corresponds to k=2,3
    
    # This decomposition is based on Larsen's rank 14 algorithm for 2x2x4.
    # I am re-using the specific coefficients from the provided "CURRENT PROGRAM" 
    # which are known to be valid.
    
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}
