import numpy as np

N = 130

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient descent to minimize Tr(G^2)
    # This minimizes the sum of squared pairwise cosines, promoting uniformity.
    
    steps = 15000
    lr_init = 0.02
    lr_decay = 0.0005
    
    for k in range(steps):
        G = P @ P.T
        grad = 4.0 * (G @ P)
        
        # Learning rate decay
        lr = lr_init * np.exp(-lr_decay * k)
        
        # Update
        P = P - lr * grad
        
        # Normalize to unit sphere
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        P = P / norms
        
    return P
