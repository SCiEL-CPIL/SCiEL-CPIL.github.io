        for p in starts:
            if time.time() > _TIME_DEADLINE:
                break
            x0 = np.concatenate([p.ravel(), [0.0, 0.0, 0.0]])
