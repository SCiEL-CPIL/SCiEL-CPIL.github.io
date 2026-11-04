<<<<<<< SEARCH
    # Verify the decomposition
    diff = _verify_decomposition(U_final, V_final, W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
=======
    # Verify the decomposition
    diff = _verify_decomposition(U_final, V_final, W_final)
    
    # Ensure arrays are of correct type and shape
    U_out = np.array(U_final, dtype=np.float64)
    V_out = np.array(V_final, dtype=np.float64)
    W_out = np.array(W_final, dtype=np.float64)
    
    return {"U": U_out, "V": V_out, "W": W_out}
>>>>>>> REPLACE
