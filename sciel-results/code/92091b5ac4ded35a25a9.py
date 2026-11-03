<<<<<<< SEARCH
        for circles0 in inits:
            if time.time() > deadline:
                break
            circles, val = _optimize(circles0, w, h, deadline, beta=100.0, max_iter=250)
            if circles is not None:
                # Polish to squeeze out more radius
                circles = _polish(circles, w, h, deadline)
                if _strictly_valid(circles, w, h):
                    val = float(np.sum(circles[:, 2]))
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
                        best_w, best_h = w, h
=======
        for circles0 in inits:
            if time.time() > deadline:
                break
            circles, val = _optimize(circles0, w, h, deadline, beta=100.0, max_iter=250)
            if circles is not None:
                # Polish to squeeze out more radius
                circles = _polish(circles, w, h, deadline)
                if _strictly_valid(circles, w, h):
                    val = float(np.sum(circles[:, 2]))
                    # Try local refine for extra gain
                    if time.time() < deadline - 5:
                        refined = _local_refine(circles, w, h, deadline - 5, max_iter=100)
                        if _strictly_valid(refined, w, h):
                            rval = float(np.sum(refined[:, 2]))
                            if rval > val:
                                circles = refined
                                val = rval
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
                        best_w, best_h = w, h
>>>>>>> REPLACE
