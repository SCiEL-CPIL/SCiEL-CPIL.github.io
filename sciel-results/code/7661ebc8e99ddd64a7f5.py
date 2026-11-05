<<<<<<< SEARCH
    def reference_integral(z_max, w0, wa, n_points=10000):
        """High- precision reference using fine trapezoidal rule"""
        z_grid = np.linspace(0, z_max, n_points)
        a_grid = 1.0 / (1.0 + z_grid)
        E_grid = E_of_a(a_grid, w0, wa)
        f_grid = 1.0 / E_grid
        # Handle singularity at z=0 if needed (E(1)=1 so f(1)=1, fine)
        return np.trapz(f_grid, z_grid)
    
    # Define a set of test cosmologies (w0, wa) pairs
    # Cover a range of dark energy models
    test_models = [
        (0.0, 0.0),      # Lambda CDM
        (-0.5, 0.0),     # Quintessence
        (-1.0, 1.0),     # CPL standard
        (-1.0, -1.0),    # Phantom
        (-0.5, 0.5),     
        (-1.5, 0.0),     # Phantom
        (-1.0, 0.5),     
        (-0.8, -0.5),    
        (-1.2, 0.3),     
        (-0.7, 0.2),     
    ]
=======
    def reference_integral(z_max, w0, wa, n_points=20000):
        """High-precision reference using fine trapezoidal rule"""
        z_grid = np.linspace(0, z_max, n_points)
        a_grid = 1.0 / (1.0 + z_grid)
        E_grid = E_of_a(a_grid, w0, wa)
        f_grid = 1.0 / E_grid
        # Handle singularity at z=0 if needed (E(1)=1 so f(1)=1, fine)
        # Use np.trapezoid for NumPy 2.0+ compatibility
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(f_grid, z_grid)
        else:
            return np.trapz(f_grid, z_grid)
    
    # Define a set of test cosmologies (w0, wa) pairs
    # Cover a range of dark energy models
    test_models = [
        (0.0, 0.0),      # Lambda CDM
        (-0.5, 0.0),     # Quintessence
        (-1.0, 1.0),     # CPL standard
        (-1.0, -1.0),    # Phantom
        (-0.5, 0.5),     
        (-1.5, 0.0),     # Phantom
        (-1.0, 0.5),     
        (-0.8, -0.5),    
        (-1.2, 0.3),     
        (-0.7, 0.2),
        (-1.3, 0.8),
        (-0.9, -0.2),
        (-0.2, 0.4),
        (-1.1, 0.1),
    ]
>>>>>>> REPLACE

