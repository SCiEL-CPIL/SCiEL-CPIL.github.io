    cd = crowding_distance(R_F, front)
    # cd is indexed 0..len(front)-1
    # We want to sort 'front' (list of global indices) by cd descending
    # Create a list of (global_idx, cd_value)
    pairs = [(front[i], cd[i]) for i in range(len(front))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    next_P_indices.extend([p[0] for p in pairs[:remaining]])
    