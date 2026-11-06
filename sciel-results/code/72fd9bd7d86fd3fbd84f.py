<<<<<<< SEARCH
    def find_spot(bin_placements, w, h, d):
        """Find the best spot for a box in a bin using frontier points."""
        # Generate candidate positions from existing placements
        candidates = set([(0, 0, 0)])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_placements, x, y, z, w, h, d):
                    valid.append((x, y, z))
        
        if not valid:
            return None
        
        # Sort by (z, y, x) to prefer lower positions (better packing)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid[0]
=======
    def find_spot(bin_placements, w, h, d):
        """Find the best spot for a box in a bin using frontier points."""
        # Generate candidate positions: all combinations of boundary coordinates
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        for (px, py, pz, pw, ph, pd) in bin_placements:
            x_coords.add(px + pw)
            y_coords.add(py + ph)
            z_coords.add(pz + pd)
        
        candidates = set()
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    if x + w <= W and y + h <= H and z + d <= D:
                        candidates.add((x, y, z))
        
        if not candidates:
            return None
        
        # Filter valid candidates
        valid = []
        for (x, y, z) in candidates:
            if can_place(bin_placements, x, y, z, w, h, d):
                valid.append((x, y, z))
        
        if not valid:
            return None
        
        # Sort by (z, y, x) to prefer lower positions (better packing)
        valid.sort(key=lambda p: (p[2], p[1], p[0]))
        return valid[0]
>>>>>>> REPLACE

