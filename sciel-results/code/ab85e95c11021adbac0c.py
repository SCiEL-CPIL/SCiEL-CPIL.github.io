<<<<<<< SEARCH
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    n_constraints, n_items = prizes.shape
=======
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Robustly enforce shape (5, 100)
    if prizes.size == 500:
        prizes = prizes.reshape(5, 100)
    if weights.size == 500:
        weights = weights.reshape(5, 100)
        
    # Handle transposed inputs if they slipped through
    if prizes.shape == (100, 5):
        prizes = prizes.T
    if weights.shape == (100, 5):
        weights = weights.T

    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    n_constraints, n_items = prizes.shape
>>>>>>> REPLACE
