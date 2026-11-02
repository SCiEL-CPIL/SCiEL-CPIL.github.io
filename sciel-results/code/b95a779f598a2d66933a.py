def _min_side_for_rotation(rot):
    verts = _all_vertices(rot)
    # Normals for standard hexagon (vertex at 0) are at 30, 90, 150 deg
    a1 = math.pi / 6.0
    a2 = math.pi / 2.0
    a3 = 5.0 * math.pi / 6.0
    
    w1 = max(v[0]*math.cos(a1) + v[1]*math.sin(a1) for v in verts) - \
         min(v[0]*math.cos(a1) + v[1]*math.sin(a1) for v in verts)
    w2 = max(v[0]*math.cos(a2) + v[1]*math.sin(a2) for v in verts) - \
         min(v[0]*math.cos(a2) + v[1]*math.sin(a2) for v in verts)
    w3 = max(v[0]*math.cos(a3) + v[1]*math.sin(a3) for v in verts) - \
         min(v[0]*math.cos(a3) + v[1]*math.sin(a3) for v in verts)
         
    return max(w1, w2, w3) / _SQ3
