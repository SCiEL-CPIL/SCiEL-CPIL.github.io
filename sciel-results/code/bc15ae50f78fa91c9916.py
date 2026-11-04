<<<<<<< SEARCH
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
=======
    # Final verification and cleanup
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Ensure we return valid floats
    if not verify(U_out, V_out, W_out):
        # Fallback to Strassen or schoolbook
        U_fb, V_fb, W_fb, rank_fb = strassen_rank50()
        if verify(U_fb, V_fb, W_fb):
            U_out, V_out, W_out = U_fb, V_fb, W_fb
        else:
            U_out, V_out, W_out, rank_fb = schoolbook()
    
    U_out = U_out.tolist()
    V_out = V_out.tolist()
    W_out = W_out.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
>>>>>>> REPLACE
