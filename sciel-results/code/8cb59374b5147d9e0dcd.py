<<<<<<< SEARCH
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() - start_time > 140:
        return None
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
