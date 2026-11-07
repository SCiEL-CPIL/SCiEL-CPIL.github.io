    # After loop
    if time.time() > DEADLINE:
         # Check for zeros
         for t in range(horizon):
             for z in range(N_z):
                 mask = consumption[t, z, :] <= 0
                 if np.any(mask):
                     consumption[t, z, mask] = m_grid[mask] # Fallback to c=m
