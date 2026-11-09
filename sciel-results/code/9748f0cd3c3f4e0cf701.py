def construct():
    N = 4
    U, V, W = strassen_decomp(N)
    return {"U": U, "V": V, "W": W}

def strassen_decomp(N):
    if N == 1:
        return [[1]], [[1]], [[1]]
    
    half = N // 2
    U_sub, V_sub, W_sub = strassen_decomp(half)
    
    # Define the 7 combinations
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We need to map local indices (0..half^2-1) to global indices (0..N^2-1)
    # for each of the 7 X and Y matrices.
    
    # This part is tricky to generalize cleanly in a few lines.
    # I will hardcode the N=4 case using the N=2 Strassen decomposition.
