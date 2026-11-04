<<<<<<< SEARCH
def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            
            if overlap:
                return False
    
    return True
=======
def _overlap_penalty(cx_arr, cy_arr, angle_arr):
    """Compute smooth overlap penalty between all hexagon pairs."""
    n = len(cx_arr)
    penalty = 0.0
    
    # Quick distance check first
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < (2.5 * MIN_DIST_CIRCLE) ** 2:
                # More detailed check needed
                vi = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
                vj = _hex_vertices(cx_arr[j], cy_arr[j], angle_arr[j])
                
                ai = _axes_from_polygon(vi)
                aj = _axes_from_polygon(vj)
                
                max_overlap = 0.0
                for axis in ai + aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    overlap = min(maxA, maxB) - max(minA, minB)
                    if overlap > max_overlap:
                        max_overlap = overlap
                
                if max_overlap > 0:
                    penalty += max_overlap ** 2
    
    return penalty
>>>>>>> REPLACE

