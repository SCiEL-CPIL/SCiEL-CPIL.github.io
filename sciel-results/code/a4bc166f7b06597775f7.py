<<<<<<< SEARCH
h_const = np.ones(N)
    # ... normalize
    conv = np.convolve(h_const, h_const, mode='full')
    score = DELTA * np.max(conv)
    best_h = h_const
    best_score = score
=======
h_const = np.ones(N)
# ... normalize
conv = np.convolve(h_const, h_const, mode='full')
score = DELTA * np.max(conv)
best_h = h_const
best_score = score
>>>>>>> REPLACE
