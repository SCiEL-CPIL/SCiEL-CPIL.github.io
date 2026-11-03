def build_4x4x4():
    # ... (same as before)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((28, 49)) # 4*7
    W4 = np.zeros((28, 49)) # 7*4
    
    # ... fill V4 and W4
    # V4[j*4+k, col] -> j in 0..3, k in 0..3.
    # But V index is j*7+k_global.
    # So V4[j*7+k, col] for k in 0..3.
    
    # W4[k*4+i, col] -> k in 0..3.
    # But W index is k*4+i? No, W index is k*n+i. n=4.
    # So W4[k*4+i, col] for k in 0..3.
    
    return U4, V4, W4
