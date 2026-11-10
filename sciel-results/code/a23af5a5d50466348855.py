        centers = generate_topology(t_name)
        # Scale centers to ensure strict separation required by validator
        centers = ensure_separation(centers, eps=1e-5)
