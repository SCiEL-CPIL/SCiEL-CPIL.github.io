        def move_point(points, idx, new_pt, area_counts):
            # points is a list of tuples
            old_pt = points[idx]
            if old_pt == new_pt: return 0
            
            # Triangles involving idx
            # i, j, k where one is idx
            # We need to subtract old areas and add new areas
            
            # Get all other points
            others = [points[j] for j in range(30) if j != idx]
            
            # Subtract old contributions
            for i in range(29):
                for j in range(i+1, 29):
                    p1 = others[i]
                    p2 = others[j]
                    area = abs(cross(old_pt, p1, p2))
                    if area > 0:
                        area_counts[area] -= 1
                        if area_counts[area] == 0:
                            del area_counts[area]
            
            # Add new contributions
            for i in range(29):
                for j in range(i+1, 29):
                    p1 = others[i]
                    p2 = others[j]
                    area = abs(cross(new_pt, p1, p2))
                    if area > 0:
                        area_counts[area] = area_counts.get(area, 0) + 1
            
            points[idx] = new_pt
        