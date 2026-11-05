<<<<<<< SEARCH
    return best_points
=======
    # Final polish: try all single-point moves from best
    if best_points and time.time() < deadline - 5:
        current_points = [p[:] for p in best_points]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(16):
                x, y = current_points[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_points = [p[:] for p in current_points]
                        new_points[i][0] = x + dx
                        new_points[i][1] = y + dy
                        new_score = _evaluate(new_points)
                        if new_score > current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
    
    return best_points
>>>>>>> REPLACE

