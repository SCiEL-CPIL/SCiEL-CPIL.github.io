<<<<<<< SEARCH
    return {"U": U, "V": V, "W": W}
=======
    # Final check on shapes
    assert U.shape == (nm, rank), f"U shape {U.shape} expected {(nm, rank)}"
    assert V.shape == (mp, rank), f"V shape {V.shape} expected {(mp, rank)}"
    assert W.shape == (pn, rank), f"W shape {W.shape} expected {(pn, rank)}"
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
