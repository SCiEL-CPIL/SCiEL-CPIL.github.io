<<<<<<< SEARCH
    def compute_L_and_penalty(params):
        """Compute (L, penalty) for given parameters."""
        verts_list = []
        all_pts = []
        for i in range(N):
            x, y, th = params[3*i], params[3*i+1], params[3*i+2]
            v = get_hex_verts_np(x, y, th)
            verts_list.append(v)
            all_pts.append(v)

        all_pts = np.vstack(all_pts)
        projections = all_pts @ outer_normals.T
        L = float(projections.max()) / (sqrt3 / 2.0)

        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                for k in range(6):
                    if point_in_convex(v1[k], v2):
                        penalty += pt_to_convex_boundary(v1[k], v2) ** 2
                    if point_in_convex(v2[k], v1):
                        penalty += pt_to_convex_boundary(v2[k], v1) ** 2
        return L, penalty
=======
    def compute_L_and_penalty(params):
        """Compute (L, penalty) for given parameters."""
        verts_list = []
        all_pts = []
        for i in range(N):
            x, y, th = params[3*i], params[3*i+1], params[3*i+2]
            v = get_hex_verts_np(x, y, th)
            verts_list.append(v)
            all_pts.append(v)

        all_pts = np.vstack(all_pts)
        projections = all_pts @ outer_normals.T
        L = float(projections.max()) / (sqrt3 / 2.0)

        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Check if any vertex of v1 is inside v2 or vice versa
                # We use a soft penalty based on penetration depth
                for k in range(6):
                    if point_in_convex(v1[k], v2):
                        d = pt_to_convex_boundary(v1[k], v2)
                        penalty += d * d
                    if point_in_convex(v2[k], v1):
                        d = pt_to_convex_boundary(v2[k], v1)
                        penalty += d * d
        return L, penalty
>>>>>>> REPLACE

