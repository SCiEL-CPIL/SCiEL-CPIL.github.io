<<<<<<< SEARCH
        best_obj = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(500):
            if time.time() > deadline - 25:
                break
=======
        best_obj = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(50):
            if time.time() > deadline - 25:
                break
>>>>>>> REPLACE
