def _objective_and_grad(c, Y, w):
    f = np.tensordot(c, Y, axes=(0, 0))
    # f shape: (70, 130)
    
    f2 = f * f
    f4 = f2 * f2
    
    sum_f2_w = np.sum(f2 * w)
    sum_f4_w = np.sum(f4 * w)
    
    if sum_f2_w < 1e-30:
        return 1e10, np.zeros_like(c)
        
    N2 = np.sqrt(sum_f2_w)
    N4 = np.sum(f4 * w)**0.25
    
    val = N4 / N2
    
    # Gradients
    # g2 = sum(f * Y * w) for each m
    # g4 = sum(f^3 * Y * w) for each m
    
    # Efficient computation:
    # f * w is (70, 130). Y is (21, 70, 130).
    # We want sum over last two axes of (f * w) * Y.
    # This is equivalent to tensordot(f*w, Y, axes=([1,2],[1,2]))? No.
    # Y[m, i, j]. We want sum_{i,j} f[i,j] w[i,j] Y[m, i, j].
    # This is a matrix multiplication if we reshape.
    # But tensordot is fine.
    
    fw = f * w
    f3w = f**3 * w
    
    g2 = np.tensordot(Y, fw, axes=([1, 2], [0, 1])) # shape (21,)
    g4 = np.tensordot(Y, f3w, axes=([1, 2], [0, 1])) # shape (21,)
    
    # grad = (1/(N2 * N4^3)) * g4 - (N4 / N2^3) * g2
    term1 = g4 / (N2 * N4**3)
    term2 = (N4 / N2**3) * g2
    
    grad = term1 - term2
    
    return val, grad
