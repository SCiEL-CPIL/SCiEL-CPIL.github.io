        # ... inside the layout loop ...
        
        # Check for overlaps
        angles_rad = [math.radians(inner_angle)] * N
        if not _check_overlaps(positions, angles_rad, tol=1e-8):
            continue # Skip invalid layout
            
        # ... proceed to find enclosing hex ...
