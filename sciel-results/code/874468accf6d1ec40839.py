    def check_overlap_fast(pos, ang_rad):
        # Precompute centers
        # pos is (N, 2)
        for i in range(N):
            c1, s1 = math.cos(ang_rad[i]), math.sin(ang_rad[i])
            v1x = pos[i, 0] + _UNIT_HEX_VERTS[:, 0] * c1 - _UNIT_HEX_VERTS[:, 1] * s1
            v1y = pos[i, 1] + _UNIT_HEX_VERTS[:, 0] * s1 + _UNIT_HEX_VERTS[:, 1] * c1
            en1x = _EDGE_NORMALS[:, 0] * c1 - _EDGE_NORMALS[:, 1] * s1
            en1y = _EDGE_NORMALS[:, 0] * s1 + _EDGE_NORMALS[:, 1] * c1
            
            for j in range(i+1, N):
                # Fast reject
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                if dx*dx + dy*dy > 4.0: # (1+1)^2
                    continue
                
                c2, s2 = math.cos(ang_rad[j]), math.sin(ang_rad[j])
                v2x = pos[j, 0] + _UNIT_HEX_VERTS[:, 0] * c2 - _UNIT_HEX_VERTS[:, 1] * s2
                v2y = pos[j, 1] + _UNIT_HEX_VERTS[:, 0] * s2 + _UNIT_HEX_VERTS[:, 1] * c2
                en2x = _EDGE_NORMALS[:, 0] * c2 - _EDGE_NORMALS[:, 1] * s2
                en2y = _EDGE_NORMALS[:, 0] * s2 + _EDGE_NORMALS[:, 1] * c2
                
                overlap = True
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        overlap = False
                        break
                if not overlap: continue
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        overlap = False
                        break
                if not overlap: continue
                
                if overlap:
                    return True
        return False
