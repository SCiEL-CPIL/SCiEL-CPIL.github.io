    # Collect edge midpoints
    edge_vecs = []
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(verts[i] - verts[j])
            # The edge length of a unit icosahedron is approx 1.0515
            # The next shortest distance is approx 1.7013 (diagonal)
            if dist < 1.2:
                mid = (verts[i] + verts[j]) / 2
                norm = np.linalg.norm(mid)
                if norm > 1e-6:
                    edge_vecs.append(mid / norm)
                    
    # We should have 30 vectors
    if len(edge_vecs) != 30:
        # Fallback: if something went wrong, we can't easily proceed.
        # But for a standard icosahedron, this should always be 30.
        pass
