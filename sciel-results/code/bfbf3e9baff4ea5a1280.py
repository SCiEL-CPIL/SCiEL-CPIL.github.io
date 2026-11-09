    def _generate_mackay_fragment(n_atoms, spacing=1.1, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
            
        # Generate full 55-atom Mackay
        full = _generate_mackay_55(spacing)
        
        if n_atoms == 55:
            return full
            
        if n_atoms < 55:
            # Indices:
            # 0: Center
            # 1-12: Vertices
            # 13-42: Edges (30)
            # 43-54: Faces (12)
            
            if n_atoms <= 13:
                return full[:n_atoms]
                
            if n_atoms <= 43:
                # Keep Center + Vertices (13)
                # Need n_atoms - 13 from Edges (30)
                n_edges_needed = n_atoms - 13
                edge_indices = list(range(13, 43))
                selected_edges = rng.choice(edge_indices, size=n_edges_needed, replace=False)
                selected_indices = list(range(13)) + list(selected_edges)
                selected_indices.sort()
                return full[selected_indices]
                
            else:
                # Keep Center + Vertices + Edges (43)
                # Need n_atoms - 43 from Faces (12)
                n_faces_needed = n_atoms - 43
                face_indices = list(range(43, 55))
                selected_faces = rng.choice(face_indices, size=n_faces_needed, replace=False)
                selected_indices = list(range(43)) + list(selected_faces)
                selected_indices.sort()
                return full[selected_indices]
                
        else:
            # n_atoms > 55
            # Add random atoms
            extra = n_atoms - 55
            # Place on sphere of radius ~ 2.5 * spacing
            radius = spacing * 2.5
            extra_points = _fibonacci_sphere(extra, radius, rng)
            return np.vstack([full, extra_points])
    