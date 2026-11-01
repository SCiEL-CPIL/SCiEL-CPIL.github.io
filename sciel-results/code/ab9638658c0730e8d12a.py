def penalty_function(params, outer_verts):
    total_penalty = 0.0
    inner_verts_list = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        viol = inside_violation(verts, outer_verts)
        if viol > 0:
            total_penalty += viol * viol
            
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            if pd > 0:
                total_penalty += pd * pd
                
    return total_penalty
