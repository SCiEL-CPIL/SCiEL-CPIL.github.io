<<<<<<< SEARCH
    # Verify with multiple seeds for robustness (vectorized, fast)
    if _verify_decomposition(U, V, W, n, m, p, seeds=(42, 123, 777)):
        return {"U": U, "V": W, "W": W}
=======
    # Verify with multiple seeds for robustness (vectorized, fast)
    if _verify_decomposition(U, V, W, n, m, p, seeds=(42, 123, 777)):
        return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
