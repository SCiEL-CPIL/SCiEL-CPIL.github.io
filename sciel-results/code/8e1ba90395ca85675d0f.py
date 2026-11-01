            centers_arr = np.array(centers)
            neighbors = get_delaunay_neighbors(centers_arr)
            
            total_penalty = 0.0
            for i, j in neighbors:
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                if pd > 0:
                    total_penalty += pd * pd
