        # 3. Spectral (Eigenvectors of random SPD matrix)
        # Generate random symmetric matrix
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        # Make it positive definite
        A = A @ A.T + np.eye(N) * 0.1
        # Eigenvectors
        eigvecs = np.linalg.eig(A)[1]
        # Take first N? We have N eigenvectors.
        # Project to 3D? The directive says "representing sphere centers as top N eigenvectors".
        # This implies the coordinates are the components of the eigenvectors?
        # If we take the first 3 components of each eigenvector, they are in R^3.
        # Let's normalize and scale them to fit in the cube.
        spectral_centers = eigvecs[:, :3]
        # Center and scale
        spectral_centers -= np.mean(spectral_centers, axis=0)
        spectral_centers /= np.max(np.abs(spectral_centers))
        spectral_centers *= 0.4 # Fit in [-0.4, 0.4]
        inits.append(spectral_centers)
