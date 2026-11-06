def _generate_comb_candidates():
    cands = []
    # Try different splits of 30 points into 2 or 3 lines
    splits = [(15, 15), (20, 10), (10, 20), (10, 10, 10)]
    
    for split in splits:
        # Determine heights
        # We want to fit in [-30, 30]
        # If 2 lines, height can be up to 60.
        # If 3 lines, heights can be 20, 20.
        
        if len(split) == 2:
            n1, n2 = split
            for h in [1, 2, 3, 5, 10]:
                # Line 1 at y = -h/2 (if h even) or -h//2
                # Let's just put line 1 at y=0 and line 2 at y=h
                # Then shift to center
                # Points on line 1: x in range
                # To maximize pairs with distance 1, use consecutive integers.
                xs1 = list(range(n1))
                xs2 = list(range(n2))
                
                pts1 = np.array([(x, 0) for x in xs1])
                pts2 = np.array([(x, h) for x in xs2])
                
                all_pts = np.vstack([pts1, pts2])
                
                # Center
                min_x = np.min(all_pts[:, 0])
                max_x = np.max(all_pts[:, 0])
                min_y = np.min(all_pts[:, 1])
                max_y = np.max(all_pts[:, 1])
                
                # Try to fit in [-30, 30]
                # Scale if needed? No, let's just translate.
                # If width > 60, we can't fit without scaling down or reducing n.
                # With n=15, width is 14. Fits easily.
                
                # Translate to center
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                all_pts -= np.array([cx, cy])
                
                # Round to integer? The task requires integer coordinates.
                # If h is integer and xs are integers, coords are integers.
                # But centering might make them non-integer.
                # We should generate integer points directly.
                
                # Regenerate with integer centering
                # Let's place line 1 at y=0, x=0..n1-1
                # Line 2 at y=h, x=0..n2-1
                # Then shift by integer dx, dy.
                
                base_pts = np.vstack([
                    np.array([(x, 0) for x in range(n1)]),
                    np.array([(x, h) for x in range(n2)])
                ])
                
                # Try a few translations
                for dx in range(-10, 11, 5):
                    for dy in range(-10, 11, 5):
                        p = base_pts + np.array([dx, dy])
                        if _inbounds(p) and _is_distinct(p):
                            cands.append(p.copy())
                            
        elif len(split) == 3:
            n1, n2, n3 = split
            for h1 in [1, 2, 5]:
                for h2 in [1, 2, 5]:
                    pts1 = np.array([(x, 0) for x in range(n1)])
                    pts2 = np.array([(x, h1) for x in range(n2)])
                    pts3 = np.array([(x, h1+h2) for x in range(n3)])
                    
                    base_pts = np.vstack([pts1, pts2, pts3])
                    
                    for dx in range(-10, 11, 5):
                        for dy in range(-10, 11, 5):
                            p = base_pts + np.array([dx, dy])
                            if _inbounds(p) and _is_distinct(p):
                                cands.append(p.copy())
                                
    return cands
