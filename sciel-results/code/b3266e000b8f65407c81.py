<<<<<<< SEARCH
            # If we got close, try to refine
            U_exp = U[:, np.newaxis, :]
            V_exp = V[np.newaxis, :, :]
            M = U_exp * V_exp
            result = M @ W.T
            if np.linalg.norm(result - T, 'fro') < 1e-5:
                # Try to snap to exact values
                if verify_fast(U, V, W, tol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
=======
            # If we got close, try to refine
            U_exp = U[:, np.newaxis, :]
            V_exp = V[np.newaxis, :, :]
            M = U_exp * V_exp
            result = M @ W.T
            if np.linalg.norm(result - T) < 1e-5:
                # Try to snap to exact values
                if verify_fast(U, V, W, tol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
>>>>>>> REPLACE

