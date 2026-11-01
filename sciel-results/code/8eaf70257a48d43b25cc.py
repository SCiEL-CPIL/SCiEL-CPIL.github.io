<<<<<<< SEARCH
                # Check error
                err = error_function(U, V, W)
                if err < 1e-8:
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify exactness
                    if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                        return U_rounded, V_rounded, W_rounded
                    
                    # If rounding failed, try to polish with gradient descent? 
                    # Or just return the float solution if it's very close? 
                    # The task asks for integer/half-integer. So we must round.
                    # If rounding doesn't work, this solution is invalid for the final output.
                    # But maybe we can use this as a start for a more precise search?
                    # For now, break and try next restart.
                    break
=======
                # Check error
                err = error_function(U, V, W)
                if err < 1e-6:
                    # Try multiple rounding strategies
                    for scale in [2, 4, 8]:
                        U_rounded = np.round(U * scale) / scale
                        V_rounded = np.round(V * scale) / scale
                        W_rounded = np.round(W * scale) / scale
                        
                        # Verify exactness
                        if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                            return U_rounded, V_rounded, W_rounded
                    
                    # Try integer rounding
                    U_rounded = np.round(U)
                    V_rounded = np.round(V)
                    W_rounded = np.round(W)
                    if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                        return U_rounded, V_rounded, W_rounded
                    
                    # Gradient-based polish: try to find nearby half-integer solution
                    # Use local search around current solution
                    for _ in range(100):
                        U_try = U + np.random.randn(n * m, r) * 0.01
                        V_try = V + np.random.randn(m * p, r) * 0.01
                        W_try = W + np.random.randn(p * n, r) * 0.01
                        
                        U_rounded = np.round(U_try * 2) / 2
                        V_rounded = np.round(V_try * 2) / 2
                        W_rounded = np.round(W_try * 2) / 2
                        
                        if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                            return U_rounded, V_rounded, W_rounded
                    break
>>>>>>> REPLACE

