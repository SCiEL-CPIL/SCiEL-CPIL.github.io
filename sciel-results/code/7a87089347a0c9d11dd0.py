<<<<<<< SEARCH
                # 2 * area = |cross product| = |xj-xi)(yk-yi) - (yj-yi)(xk-xi)|
                det = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
=======
                # 2 * area = |cross product| = |(xj-xi)(yk-yi) - (yj-yi)(xk-xi)|
                xk, yk = points[k]
                det = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
>>>>>>> REPLACE
