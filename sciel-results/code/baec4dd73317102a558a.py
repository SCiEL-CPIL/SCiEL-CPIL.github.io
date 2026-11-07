            if used > 0:
                x[1] = float(x2)
                f = problem(x)
                f = np.array(f, dtype=float)
                points.append(x.copy())
                objectives.append(f)
                recorded_objs.append(f)
                used += 1
                remaining_candidates.remove(best_idx)
            else:
