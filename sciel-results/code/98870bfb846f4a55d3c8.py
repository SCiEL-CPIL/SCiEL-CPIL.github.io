    # Ensure final output is lists
    if not isinstance(best_U, list):
        best_U = best_U.tolist()
    if not isinstance(best_V, list):
        best_V = best_V.tolist()
    if not isinstance(best_W, list):
        best_W = best_W.tolist()
        
    return {"U": best_U, "V": best_V, "W": best_W}
