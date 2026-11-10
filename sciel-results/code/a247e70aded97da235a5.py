<<<<<<< SEARCH
        # Vectorized computation
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # For normal incidence, s and p polarizations are the same
        
        # We'll compute the total characteristic matrix M = M_1 * M_2 * ... * M_N
        # Then Y = M_11 / M_21 (admittance)
        # R = |Y - n_0| / (Y + n_0) squared... wait, let me be more careful.
        
        # Actually, the characteristic matrix approach:
        # For each layer i:
        #   M_i = [[cos(delta_i), (i/n_i)sin(delta_i)], [i*n_i*sin(delta_i), cos(delta_i)]]
        # Total M = M_1 * M_2 * ... * M_N
        # B = M_11 * n_s + i * M_21 * n_s^2 ... no wait
        # Actually: B = M_11 * n_s + i * M_21 * n_s^2 is not right either.
        
        # Let me use the standard formulation:
        # The characteristic matrix for layer i is:
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i/lambda
        
        # The total characteristic matrix is M = M_1 * M_2 * ... * M_N
        # Then the admittance at the input side is:
        # Y = (M_11 * n_s + i * M_21 * n_s^2) / (M_12 * n_s + i * M_22 * n_s^2)
        
        # Hmm, actually I think the correct formula is:
        # B = M_11 * n_s + i * M_21 * n_s^2
        # C = M_12 * n_s + i * M_22 * n_s^2
        # Y = B/C
        # R = |(Y - n_0)/(Y + n_0)|^2
        
        # Let me verify with a single layer:
        # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # B = cos(delta)*n_s + i*(i*n*sin(delta))*n_s^2 = cos(delta)*n_s - n*sin(delta)*n_s^2
        # C = (i*sin(delta)/n)*n_s + i*cos(delta)*n_s^2 = i*sin(delta)*n_s/n + i*cos(delta)*n_s^2
        # Y = B/C
        # For quarter wave: delta = pi/2, cos=0, sin=1
        # B = -n*n_s^2
        # C = i*n_s/n + i*n_s^2 = i*n_s(1/n + n_s)
        # Y = -n*n_s^2 / (i*n_s*(1/n + n_s)) = -n*n_s / (i*(1/n + n_s)) = i*n*n_s / (1/n + n_s)
        # Y = i*n*n_s / ((n + n_s*n)/n) = i*n^2*n_s / (n + n_s*n) ... this is getting messy
        
        # Let me just use a simpler approach: compute it numerically for each wavelength.
        
        reflectances = np.zeros(len(wavelengths))
        
        for idx, lam in enumerate(wavelengths):
            # Compute phase for each layer
            delta = 2.0 * np.pi * n_layers_arr * d_arr / lam
            
            # Build characteristic matrix
            # Start with identity
            M11 = 1.0 + 0j
            M12 = 0.0 + 0j
            M21 = 0.0 + 0j
            M22 = 1.0 + 0j
            
            for i in range(n_layers):
                cos_d = np.cos(delta[i])
                sin_d = np.sin(delta[i])
                n_i = n_layers_arr[i]
                
                # Layer characteristic matrix
                m11 = cos_d
                m12 = 1j * sin_d / n_i
                m21 = 1j * n_i * sin_d
                m22 = cos_d
                
                # Multiply M = M * m
                new_M11 = M11 * m11 + M12 * m21
                new_M12 = M11 * m12 + M12 * m22
                new_M21 = M21 * m11 + M22 * m21
                new_M22 = M21 * m12 + M22 * m22
                
                M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
            
            # Compute admittance
            B = M11 * n_s + 1j * M21 * n_s ** 2
            C = M12 * n_s + 1j * M22 * n_s ** 2
            Y = B / C
            
            # Reflectance
            r = (Y - n_0) / (Y + n_0)
            reflectances[idx] = np.abs(r) ** 2
        
        return reflectances
=======
        # Vectorized computation across all wavelengths
        # delta = 2*pi*n*d/lambda for each layer and wavelength
        # Shape: (n_layers, n_wavelengths)
        delta = 2.0 * np.pi * np.outer(n_layers_arr, d_arr) / wavelengths[np.newaxis, :]
        
        cos_d = np.cos(delta)  # (n_layers, n_wavelengths)
        sin_d = np.sin(delta)  # (n_layers, n_wavelengths)
        
        # Initialize total characteristic matrix as identity for each wavelength
        # M11, M12, M21, M22 each shape (n_wavelengths,)
        M11 = np.ones(len(wavelengths), dtype=complex)
        M12 = np.zeros(len(wavelengths), dtype=complex)
        M21 = np.zeros(len(wavelengths), dtype=complex)
        M22 = np.ones(len(wavelengths), dtype=complex)
        
        for i in range(n_layers):
            n_i = n_layers_arr[i]
            # Layer characteristic matrix components (shape: n_wavelengths)
            m11 = cos_d[i]
            m12 = 1j * sin_d[i] / n_i
            m21 = 1j * n_i * sin_d[i]
            m22 = cos_d[i]
            
            # Multiply M = M * m
            new_M11 = M11 * m11 + M12 * m21
            new_M12 = M11 * m12 + M12 * m22
            new_M21 = M21 * m11 + M22 * m21
            new_M22 = M21 * m12 + M22 * m22
            
            M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
        # Compute admittance
        B = M11 * n_s + 1j * M21 * n_s ** 2
        C = M12 * n_s + 1j * M22 * n_s ** 2
        Y = B / C
        
        # Reflectance
        r = (Y - n_0) / (Y + n_0)
        reflectances = np.abs(r) ** 2
        
        return reflectances
>>>>>>> REPLACE

