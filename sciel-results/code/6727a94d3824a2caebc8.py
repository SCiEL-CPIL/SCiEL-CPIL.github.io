<<<<<<< SEARCH
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(val - 1.0) > 1e-10:
=======
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if bool(abs(val - 1.0) > 1e-10):
>>>>>>> REPLACE
