        C = np.zeros((n, p, m)) # Note: original used (n, p, m) but indices were i,k,j?
        # Original: C[i, k, j]?
        # Let's check original C construction:
        # C += Uk[:, r][:, None] * Wk[:, r][None, :] * Vk[:, r][None, :]
        # Uk is (nm, 1) -> (nm, 1, 1)? No, Uk[:, r] is (nm,).
        # Uk[:, r][:, None] is (nm, 1).
        # Wk[:, r][None, :] is (1, pn).
        # Vk[:, r][None, :] is (1, mp).
        # This broadcasting is wrong.
        # (nm, 1) * (1, pn) -> (nm, pn).
        # Then * (1, mp) -> (nm, pn, mp)? No, dimensions don't align for simple broadcast to 3D.
        # The original code for C construction is likely buggy too.
   