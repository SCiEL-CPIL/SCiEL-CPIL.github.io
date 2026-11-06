<<<<<<< SEARCH
    n_vars = 511
    c = np.array([-weights[1:512]]) # Minimize -sum(w_i x_i)
=======
    n_vars = 511
    c = -np.array(weights[1:512]) # Minimize -sum(w_i x_i)
>>>>>>> REPLACE
