import numpy as np

def construct():
    # 1. Strassen 2x2x2
    # U2, V2, W2 of shape (4, 7)
    # ...
    
    # 2. 4x4x4 via Kronecker
    # U4 = kron(U2, U2) reshaped to (16, 49)
    # ...
    
    # 3. 4x4x2 Schoolbook
    # U2s, V2s, W2s of shape (16, 32), (8, 32), (8, 32)
    # ...
    
    # 4. Combine
    # U = hstack(U4, U2s)
    # V = zeros((24, 81))
    # V[:16, :49] = V4
    # V[16:, 49:] = V2s
    # W = zeros((24, 81))
    # W[:16, :49] = W4
    # W[16:, 49:] = W2s
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
