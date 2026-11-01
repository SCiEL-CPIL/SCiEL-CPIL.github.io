    def calc_cost(pos, ang, outer_angle):
        normals = []
        for k in range(6):
            theta_n = outer_angle + math.pi/6.0 + k * math.pi / 3.0
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        max_proj = 0.0
        all_verts = []
        all_boxes = []
        
        for i in range(N):
            verts = get_hex_vertices(pos[i][0], pos[i][1], ang[i])
            all_verts.append(verts)
            all_boxes.append(bounding_box(verts))
            
            for v in verts:
                for n in normals:
                    proj = v[0]*n[0] + v[1]*n[1]
                    if proj > max_proj:
                        max_proj = proj
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap(all_verts[i], all_verts[j], all_boxes[i], all_boxes[j]):
                    penalty += 1000.0
                    
        return max_proj + penalty
