    pts = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2_np = np.sum(diff ** 2, axis=2)
    
    masks = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for a in range(NUM_POINTS):
        for b in range(a + 1, NUM_POINTS):
            dab = dist2_np[a, b]
            dac = dist2_np[a, :]
            dbc = dist2_np[b, :]
            is_iso = (dac == dab) | (dbc == dab) | (dac == dbc)
            is_iso[a] = False
            is_iso[b] = False
            
            # Convert boolean array to int mask
            # Find indices where True
            idxs = np.nonzero(is_iso)[0]
            mask_val = 0
            for c in idxs:
                mask_val |= (1 << c)
            
            masks[a][b] = mask_val
            masks[b][a] = mask_val
