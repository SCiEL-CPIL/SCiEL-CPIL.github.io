<<<<<<< SEARCH
        best_s = float('inf')
        for oa in [0, 30, 60, 90, 120, 150]:
             outward_normals = np.array([
=======
        best_s = float('inf')
        oa_best = 0
        for oa in [0, 30, 60, 90, 120, 150]:
            outward_normals = np.array([
>>>>>>> REPLACE

<<<<<<< SEARCH
            if s_val < best_s:
                best_s = s_val
=======
            if s_val < best_s:
                best_s = s_val
                oa_best = oa
>>>>>>> REPLACE
