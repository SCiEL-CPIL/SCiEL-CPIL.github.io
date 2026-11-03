<<<<<<< SEARCH
        # Optimize ring structure parameters
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(15.0, deadline - (time.time() - start_time) - 30)
        )
=======
        # Optimize ring structure parameters
        # DE is slower, so give it a reasonable chunk of time but ensure we have time for binary search
        time_for_de = min(10.0, max(2.0, deadline - (time.time() - start_time) - 40))
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=time_for_de
        )
>>>>>>> REPLACE
