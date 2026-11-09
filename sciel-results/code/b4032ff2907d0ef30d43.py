    def compute_min_outer_side_from_verts(verts_list):
        max_proj = 0.0
        for verts in verts_list:
            proj_matrix = np.dot(verts, outer_normals.T)
            current_max = np.max(proj_matrix)
            if current_max > max_proj:
                max_proj = current_max
        return max_proj / inradius_factor
